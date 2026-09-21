//! Normalized priors on final-function properties. Every smooth function on a
//! frozen basis splits into its constant coefficient, column zero, and its
//! variation over the other columns. Variation carries the basis's frozen
//! penalty blocks, whose strength limits leave the constant function. Levels
//! carry function-measure roots with exact determinants, or chart laws with
//! their Jacobians. The one coordinate without a normalized law is a Student-t
//! channel's intercept level, an unpenalized coefficient under the flat measure.
//! Evidence integrates it as REML integrates an unpenalized block: under that
//! flat measure, with its Laplace log-determinant term, identically in every
//! structure, so the measure's scale cancels from every Bayes factor.
//! Construction and evaluation are generic over the scalar field, so the f64
//! production route is the route tests evaluate with rounding bounds. Global
//! coefficient integration remains necessary.
use super::category_prior::{CategoryEvaluation, CategoryPriors, scaled};
use super::decoder_prior::DecoderPriorEvaluation;
use super::law::{BasisPenalty, JointHistory, JointLikelihood, MeasurementFamily, invalid, numerical};
use super::structural_prior::{ScalarPriorEvaluation, StructuralFunction, structural_priors};
use crate::chain::log_sum_exp;
use crate::scalar::{add_real, div, exp, ln, sqrt};
use crate::{EventHistoryError, MarkKind};
use gam_math::nested_dual::JetField;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::Arc;

/// One learned strength. A variation strength belongs to one frozen penalty
/// block of its basis and is shared by every function on that basis: the
/// baseline functions of every mark, the decoder logit functions of every mark
/// and signature, the drive functions of every signature, and the intercept and
/// loading functions of every channel. A level strength belongs to one family
/// of constant coefficients.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FunctionPenalty {
    Decoder { mark: usize },
    DecoderVariation { penalty: usize },
    BaselineVariation { penalty: usize },
    /// The mean drive level `(u_k + B_k' mu)^2`.
    DriveLevel,
    /// The genetic dependence `B_k' Sigma B_k` of the drive level; its limit keeps `u_k`.
    GeneticDrive,
    DriveVariation { penalty: usize },
    EntryMean,
    EntryPrevalence { mark: usize },
    DiseaseJump { mark: usize },
    MeasurementEffect { channel: usize },
    MeasurementVariation { penalty: usize },
    TemporalVariation,
    MeasurementPrecision { channel: usize },
    TailVarianceInflation { channel: usize },
    CountOverdispersion { channel: usize },
    CountMean { channel: usize },
    BaselineLevel,
}

struct FunctionRoot<S> {
    root: Array2<S>,
    log_determinant: S,
}

/// A Gaussian block's energy: `||R beta||^2` for a positive-definite measure
/// root, or `beta' S beta` for one frozen penalty block of structural rank.
enum GaussianForm<S> {
    Root(Arc<FunctionRoot<S>>),
    Penalty { local: Arc<Array2<S>>, rank: usize },
}

struct GaussianFunction<S> {
    strength: usize,
    form: GaussianForm<S>,
    coordinates: Vec<Range<usize>>,
    /// Student-t functions are measured in residual-scale units. The prior
    /// precision is exp(rho - 2*log_scale), including its normalizer.
    log_scale: Option<usize>,
}

/// Frozen function measures for one model and cohort design. The coordinate
/// order of strengths is exposed by `penalties`; inactive zero-dimensional
/// functions introduce no strength coordinates.
pub struct JointFunctionPriors<'m, S = f64> {
    model: &'m JointLikelihood,
    penalties: Vec<FunctionPenalty>,
    gaussian: Vec<GaussianFunction<S>>,
    /// The strength-free normalizer of every Gaussian block: half the log
    /// determinants of the measure roots and of each penalty range's summed
    /// blocks, counted once per function on that range.
    normalizer: S,
    /// ½ log 2π, formed over S.
    half_log_two_pi: S,
    structural: Vec<Vec<StructuralFunction>>,
    structural_strengths: Vec<usize>,
    category: CategoryPriors,
    baseline_mean_design: Vec<S>,
    log_followup_scale: S,
    decoder_strengths: usize,
    level_strength: usize,
}

struct GaussianEvaluation<S> {
    half_log_precision: S,
    weighted_penalty: S,
    /// Derivatives in each repeated function's contiguous coordinate range.
    gradients: Vec<Vec<S>>,
}

pub struct FunctionPriorEvaluation<'a, 'm, S = f64> {
    prior: &'a JointFunctionPriors<'m, S>,
    /// A Gaussian family left out of this evaluation, at its zero-effect face.
    excluded: Option<usize>,
    decoder: DecoderPriorEvaluation<S>,
    gaussian: Vec<GaussianEvaluation<S>>,
    structural: Vec<Vec<ScalarPriorEvaluation<S>>>,
    category: CategoryEvaluation<S>,
    baseline_weights: Vec<S>,
    log_density: S,
    gradient: Vec<S>,
    strength_gradient: Vec<S>,
    strength_second: Vec<S>,
}

impl<S: JetField> FunctionRoot<S> {
    /// The upper Cholesky root `R = L'` of a symmetric positive-definite Gram and
    /// `log det(Gram)`. A nonpositive or non-finite pivot refuses the basis: aliased
    /// directions are removed before fitting, never ridged.
    fn from_gram(gram: &[Vec<S>], zero: &S) -> Result<Self, EventHistoryError> {
        let n = gram.len();
        let mut lower = vec![vec![zero.clone(); n]; n];
        let mut log_determinant = zero.clone();
        for i in 0..n {
            for j in 0..=i {
                let value = (0..j).fold(gram[i][j].clone(), |v, q| {
                    v.sub(&lower[i][q].mul(&lower[j][q]))
                });
                if i == j {
                    if !(value.value().is_finite() && value.value() > 0.0) {
                        return Err(invalid(
                            "function measure does not identify an independent basis; remove aliased basis directions before fitting",
                        ));
                    }
                    log_determinant = log_determinant.add(&ln(&value));
                    lower[i][i] = sqrt(&value);
                } else {
                    lower[i][j] = div(&value, &lower[j][j]);
                }
            }
        }
        Ok(Self {
            root: Array2::from_shape_fn((n, n), |(i, j)| lower[j][i].clone()),
            log_determinant,
        })
    }
    fn identity(width: usize, zero: &S) -> Self {
        Self {
            root: Array2::from_shape_fn((width, width), |(i, j)| {
                zero.constant_like(f64::from(i == j))
            }),
            log_determinant: zero.clone(),
        }
    }
    fn quadratic(&self, coefficients: &[S], half_log_precision: &S) -> (S, Vec<S>) {
        // Split the precision between the coefficient vector and the final
        // transpose product. exp(rho) need not itself be representable.
        let zero = half_log_precision.constant_like(0.0);
        let x: Vec<S> = coefficients
            .iter()
            .map(|v| scaled(v, half_log_precision))
            .collect();
        let width = x.len();
        let transformed: Vec<S> = (0..width)
            .map(|i| {
                (0..width).fold(zero.clone(), |sum, j| sum.add(&self.root[[i, j]].mul(&x[j])))
            })
            .collect();
        let energy = transformed
            .iter()
            .fold(zero.clone(), |sum, t| sum.add(&t.mul(t)))
            .scale(0.5);
        let gradient = (0..width)
            .map(|j| {
                scaled(
                    &(0..width).fold(zero.clone(), |sum, i| {
                        sum.add(&self.root[[i, j]].mul(&transformed[i]))
                    }),
                    half_log_precision,
                )
            })
            .collect();
        (energy, gradient)
    }
}

/// `½ exp(2 half) beta' S beta` and `exp(2 half) S beta`, scaling before the
/// products so exp(rho) need not itself be representable.
fn penalty_quadratic<S: JetField>(local: &Array2<S>, coefficients: &[S], half: &S) -> (S, Vec<S>) {
    let zero = half.constant_like(0.0);
    let x: Vec<S> = coefficients.iter().map(|v| scaled(v, half)).collect();
    let width = x.len();
    let product: Vec<S> = (0..width)
        .map(|i| (0..width).fold(zero.clone(), |sum, j| sum.add(&local[[i, j]].mul(&x[j]))))
        .collect();
    let energy = x
        .iter()
        .zip(&product)
        .fold(zero.clone(), |sum, (a, b)| sum.add(&a.mul(b)))
        .scale(0.5);
    let gradient = product.iter().map(|p| scaled(p, half)).collect();
    (energy, gradient)
}

impl<S: JetField> GaussianForm<S> {
    fn quadratic(&self, coefficients: &[S], half: &S) -> (S, Vec<S>) {
        match self {
            Self::Root(root) => root.quadratic(coefficients, half),
            Self::Penalty { local, .. } => penalty_quadratic(local, coefficients, half),
        }
    }
    fn rank(&self) -> usize {
        match self {
            Self::Root(root) => root.root.ncols(),
            Self::Penalty { rank, .. } => *rank,
        }
    }
}

fn add_outer<S: JetField>(gram: &mut [Vec<S>], row: &[S], weight: &S) {
    for i in 0..row.len() {
        for j in 0..=i {
            gram[i][j] = gram[i][j].add(&row[i].mul(&row[j]).mul(weight));
            gram[j][i] = gram[i][j].clone();
        }
    }
}

fn product_root<S: JetField>(left: &FunctionRoot<S>, right: &FunctionRoot<S>) -> FunctionRoot<S> {
    let l = left.root.ncols();
    let r = right.root.ncols();
    FunctionRoot {
        root: Array2::from_shape_fn((l * r, l * r), |(p, q)| {
            left.root[[p / r, q / r]].mul(&right.root[[p % r, q % r]])
        }),
        log_determinant: left
            .log_determinant
            .scale(r as f64)
            .add(&right.log_determinant.scale(l as f64)),
    }
}

/// One frozen basis's penalty ranges, each with half the log determinant of
/// its summed blocks. Contract: no range contains the constant column 0; ranges
/// are identical or disjoint and tile the other columns; the structural ranks
/// of the blocks sharing a range sum to its width; and their sum is positive
/// definite. The blocks' ranges then form a direct sum, so
/// `det(sum_j lambda_j S_j) = prod_j lambda_j^rank_j det(sum_j S_j)` exactly and
/// every strength limit leaves the constant function.
fn penalty_ranges<S: JetField>(
    penalties: &[BasisPenalty],
    width: usize,
    zero: &S,
) -> Result<Vec<(Range<usize>, S)>, EventHistoryError> {
    let refuse = || {
        invalid(
            "basis penalties must tile the non-constant columns with ranks filling each range and a positive-definite sum",
        )
    };
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for penalty in penalties {
        let range = penalty.columns.clone();
        if range.start == 0 || range.is_empty() {
            return Err(refuse());
        }
        if !ranges.contains(&range) {
            if ranges.iter().any(|r| r.start < range.end && range.start < r.end) {
                return Err(refuse());
            }
            ranges.push(range);
        }
    }
    ranges.sort_by_key(|r| r.start);
    let mut next = 1;
    for range in &ranges {
        if range.start != next {
            return Err(refuse());
        }
        next = range.end;
    }
    if next != width.max(1) {
        return Err(refuse());
    }
    let mut out = Vec::with_capacity(ranges.len());
    for range in ranges {
        let size = range.len();
        let blocks: Vec<&BasisPenalty> = penalties.iter().filter(|p| p.columns == range).collect();
        if blocks.iter().map(|p| p.rank).sum::<usize>() != size {
            return Err(refuse());
        }
        let sum: Vec<Vec<S>> = (0..size)
            .map(|i| {
                (0..size)
                    .map(|j| {
                        blocks
                            .iter()
                            .fold(zero.clone(), |s, p| s.add(&zero.constant_like(p.local[[i, j]])))
                    })
                    .collect()
            })
            .collect();
        let determinant = FunctionRoot::from_gram(&sum, zero).map_err(|_| refuse())?;
        out.push((range, determinant.log_determinant.scale(0.5)));
    }
    Ok(out)
}

fn local_matrix<S: JetField>(penalty: &BasisPenalty, zero: &S) -> Array2<S> {
    penalty.local.mapv(|v| zero.constant_like(v))
}

fn representable<S: JetField>(out: Vec<S>, reason: &str) -> Result<Vec<S>, EventHistoryError> {
    if out.iter().any(|v| !v.value().is_finite()) {
        return Err(numerical(reason));
    }
    Ok(out)
}

impl JointLikelihood {
    /// Equal subject measure; within a follow-up, time is measured by the
    /// normalized compensator quadrature. Genetic effects use the declared
    /// Gaussian score law. The baseline, drive and population bases carry their
    /// unit constant in column zero, and their frozen penalties act on the
    /// other columns; each function family's level has its own prior.
    ///
    /// Entry prevalence is a function contrast against no prevalent diagnosis,
    /// not an invented distribution of pre-entry event-free follow-up.
    pub fn function_priors<'m>(
        &'m self,
        histories: &[&JointHistory],
    ) -> Result<JointFunctionPriors<'m>, EventHistoryError> {
        self.function_priors_over(histories, &0.0)
    }

    /// The construction over any scalar field; `zero` supplies its shape. Check-only bands
    /// evaluate the same route at `numerical::Running`.
    pub(super) fn function_priors_over<'m, S: JetField>(
        &'m self,
        histories: &[&JointHistory],
        zero: &S,
    ) -> Result<JointFunctionPriors<'m, S>, EventHistoryError> {
        if histories.is_empty() {
            return Err(invalid("function priors require a cohort design"));
        }
        let k = self.spec.signatures;
        for h in histories {
            self.validate_history(h)?;
            if h.baseline_design.column(0).iter().any(|&v| v != 1.0)
                || (k > 0 && h.drive_design.column(0).iter().any(|&v| v != 1.0))
            {
                return Err(invalid(
                    "joint function priors require unit constants in column zero of the baseline and drive bases",
                ));
            }
        }
        let constant = |v: f64| zero.constant_like(v);
        let genes = self.spec.genetic_mean.len() + 1;
        let b = self.spec.baseline_columns;
        let c = self.spec.drive_columns;
        let p = self.spec.population_columns;
        let e = self.spec.entry_columns + 1;
        // Kronecker roots and drive penalties are dense in (basis width x genetic channels)^2.
        for width in [c, e] {
            width
                .checked_mul(genes)
                .and_then(|w| w.checked_mul(w))
                .ok_or_else(|| invalid("function-prior root dimension overflow"))?;
        }
        let subjects = histories.len() as f64;
        let mut log_spans = Vec::with_capacity(histories.len());
        for h in histories {
            let span = constant(h.times[h.times.len() - 1]).sub(&constant(h.times[0]));
            if !(span.value().is_finite() && span.value() > 0.0) {
                return Err(invalid(
                    "function priors require finite positive follow-up spans",
                ));
            }
            log_spans.push(ln(&span));
        }
        let log_followup_scale = log_sum_exp(&log_spans).sub(&ln(&constant(subjects)));
        // Each subject's normalized compensator measure, formed over S from its exact weights.
        let mut baseline_mean_design = vec![zero.clone(); b];
        for h in histories {
            let measure = h
                .points
                .iter()
                .fold(zero.clone(), |sum, point| sum.add(&constant(point.weight)))
                .mul(&constant(subjects));
            for (row, point) in h.points.iter().enumerate() {
                for (j, mean) in baseline_mean_design.iter_mut().enumerate() {
                    *mean = mean.add(&div(
                        &constant(point.weight).mul(&constant(h.baseline_design[[row, j]])),
                        &measure,
                    ));
                }
            }
        }
        // Its unit intercept is an exact coordinate direction, irrespective
        // of accumulated floating-point error in normalized exposure weights.
        baseline_mean_design[0] = constant(1.0);
        let decoder_strengths = if k == 0 { 0 } else { self.spec.marks.len() };
        let mut result = JointFunctionPriors {
            model: self,
            penalties: (0..decoder_strengths)
                .map(|mark| FunctionPenalty::Decoder { mark })
                .collect(),
            gaussian: Vec::new(),
            normalizer: zero.clone(),
            half_log_two_pi: ln(&constant(2.0 * std::f64::consts::PI)).scale(0.5),
            structural: Vec::new(),
            structural_strengths: Vec::new(),
            category: CategoryPriors::new(self),
            baseline_mean_design,
            log_followup_scale,
            decoder_strengths,
            level_strength: 0,
        };
        let marks = self.spec.marks.len();
        // Baseline variation: one strength per frozen penalty block, shared by marks.
        let ranges = penalty_ranges(&self.spec.baseline_penalties, b, zero)?;
        for (index, penalty) in self.spec.baseline_penalties.iter().enumerate() {
            let local = Arc::new(local_matrix(penalty, zero));
            let coordinates = (0..marks)
                .map(|mark| {
                    let base = self.layout.baseline.start + mark * b;
                    base + penalty.columns.start..base + penalty.columns.end
                })
                .collect();
            result.push_gaussian(
                FunctionPenalty::BaselineVariation { penalty: index },
                GaussianForm::Penalty {
                    local,
                    rank: penalty.rank,
                },
                coordinates,
                None,
            );
        }
        for (_, half_log_det) in &ranges {
            result.normalizer = result.normalizer.add(&half_log_det.scale(marks as f64));
        }
        let channels = self.spec.measurements.len();
        let population = if k > 0 || channels > 0 {
            penalty_ranges(&self.spec.population_penalties, p, zero)?
        } else {
            Vec::new()
        };
        if k > 0 {
            result.add_signature_functions(histories, &population, zero)?;
        }
        // Intercept and loading variation over the population basis, in
        // residual-scale units for Student-t channels.
        for (channel, family) in self.spec.measurements.iter().enumerate() {
            let start = self.layout.measurement_location[channel].start;
            let scale = matches!(family, MeasurementFamily::StudentT)
                .then_some(self.layout.measurement_shape[channel].start);
            for (index, penalty) in self.spec.population_penalties.iter().enumerate() {
                let local = Arc::new(local_matrix(penalty, zero));
                let coordinates = (0..=k)
                    .map(|function| {
                        let base = start + function * p;
                        base + penalty.columns.start..base + penalty.columns.end
                    })
                    .collect();
                result.push_gaussian(
                    FunctionPenalty::MeasurementVariation { penalty: index },
                    GaussianForm::Penalty {
                        local,
                        rank: penalty.rank,
                    },
                    coordinates,
                    scale,
                );
            }
            for (_, half_log_det) in &population {
                result.normalizer = result.normalizer.add(&half_log_det.scale((k + 1) as f64));
            }
        }
        result.level_strength = result.penalties.len();
        result.penalties.push(FunctionPenalty::BaselineLevel);
        for (label, functions) in structural_priors(self) {
            result.structural_strengths.push(result.penalties.len());
            result.penalties.push(label);
            result.structural.push(functions);
        }
        Ok(result)
    }
}

impl<'m, S: JetField> JointFunctionPriors<'m, S> {
    /// Add one Gaussian function family under `label`, sharing that label's
    /// strength when it already exists.
    fn push_gaussian(
        &mut self,
        label: FunctionPenalty,
        form: GaussianForm<S>,
        coordinates: Vec<Range<usize>>,
        log_scale: Option<usize>,
    ) {
        let strength = match self.penalties.iter().position(|existing| *existing == label) {
            Some(index) => index,
            None => {
                self.penalties.push(label);
                self.penalties.len() - 1
            }
        };
        self.gaussian.push(GaussianFunction {
            strength,
            form,
            coordinates,
            log_scale,
        });
    }

    /// Decoder variation, genetic drive level and variation, entry, disease-jump
    /// and loading-level functions of the latent state. Roots and penalties are
    /// shared across signatures.
    fn add_signature_functions(
        &mut self,
        histories: &[&JointHistory],
        population: &[(Range<usize>, S)],
        zero: &S,
    ) -> Result<(), EventHistoryError> {
        let constant = |v: f64| zero.constant_like(v);
        let model = self.model;
        let k = model.spec.signatures;
        let marks = model.spec.marks.len();
        let genes = model.spec.genetic_mean.len() + 1;
        let c = model.spec.drive_columns;
        let p = model.spec.population_columns;
        let e = model.spec.entry_columns + 1;
        let subjects = histories.len() as f64;
        // Decoder logit variation over the population basis. The Dirichlet
        // level prior acts on column 0 of every logit function.
        for (index, penalty) in model.spec.population_penalties.iter().enumerate() {
            let local = Arc::new(local_matrix(penalty, zero));
            let coordinates = (0..marks * k)
                .map(|function| {
                    let base = model.layout.decoder.start + function * p;
                    base + penalty.columns.start..base + penalty.columns.end
                })
                .collect();
            self.push_gaussian(
                FunctionPenalty::DecoderVariation { penalty: index },
                GaussianForm::Penalty {
                    local,
                    rank: penalty.rank,
                },
                coordinates,
                None,
            );
        }
        for (_, half_log_det) in population {
            self.normalizer = self.normalizer.add(&half_log_det.scale((marks * k) as f64));
        }
        // E[(a + b'g)^2] = (a+b'mu)^2 + ||L^{-1}b||^2,
        // where L L' is genetic precision. Construct this root directly;
        // forming mu*mu' first would lose small genetic variances.
        // The genetic precision's Cholesky factor L and log det, recomputed over S from the
        // declared precision rather than read from the law's stored f64 factor.
        let precision: Vec<Vec<S>> = (0..genes - 1)
            .map(|i| {
                (0..genes - 1)
                    .map(|j| constant(model.spec.genetic_precision[[i, j]]))
                    .collect()
            })
            .collect();
        let precision_root = FunctionRoot::from_gram(&precision, zero)?;
        let lower = |i: usize, j: usize| &precision_root.root[[j, i]];
        let mut genetic_root = vec![vec![zero.clone(); genes]; genes];
        genetic_root[0][0] = constant(1.0);
        for j in 1..genes {
            genetic_root[0][j] = constant(model.spec.genetic_mean[j - 1]);
        }
        for column in 1..genes {
            for i in 1..genes {
                let value = (1..i).fold(constant(f64::from(i == column)), |v, j| {
                    v.sub(&lower(i - 1, j - 1).mul(&genetic_root[j][column]))
                });
                genetic_root[i][column] = div(&value, lower(i - 1, i - 1));
            }
        }
        let genetic = Arc::new(FunctionRoot {
            root: Array2::from_shape_fn((genes, genes), |(i, j)| genetic_root[i][j].clone()),
            log_determinant: precision_root.log_determinant.neg(),
        });
        // The drive level, column 0 of each axis's drive function u_k + B_k g, has
        // E[(a + b'g)^2] = (a + b'mu)^2 + b' Sigma b: two complementary blocks, the mean level of
        // rank one and the genetic dependence of rank `scores`, whose limit keeps u_k. Their sum
        // is M = R_g' R_g, so the range normalizer is ½ log det M per axis.
        let levels: Vec<Range<usize>> = (0..k)
            .map(|axis| {
                let start = model.layout.drive.start + axis * c * genes;
                start..start + genes
            })
            .collect();
        let mean_level = Arc::new(Array2::from_shape_fn((genes, genes), |(i, j)| {
            genetic_root[0][i].mul(&genetic_root[0][j])
        }));
        self.push_gaussian(
            FunctionPenalty::DriveLevel,
            GaussianForm::Penalty {
                local: mean_level,
                rank: 1,
            },
            levels.clone(),
            None,
        );
        if genes > 1 {
            let dependence = Arc::new(Array2::from_shape_fn((genes, genes), |(i, j)| {
                (1..genes).fold(zero.clone(), |sum, q| {
                    sum.add(&genetic_root[q][i].mul(&genetic_root[q][j]))
                })
            }));
            self.push_gaussian(
                FunctionPenalty::GeneticDrive,
                GaussianForm::Penalty {
                    local: dependence,
                    rank: genes - 1,
                },
                levels,
                None,
            );
        }
        self.normalizer = self
            .normalizer
            .add(&genetic.log_determinant.scale(0.5 * k as f64));
        // Drive variation: the roughness of u_k(t) + B_k(t) g under the genetic law
        // is beta' (S ⊗ M) beta, with M = E[(1,g)(1,g)'] = R_g' R_g.
        let moment: Vec<Vec<S>> = (0..genes)
            .map(|i| {
                (0..genes)
                    .map(|j| {
                        (0..genes).fold(zero.clone(), |s, q| {
                            s.add(&genetic.root[[q, i]].mul(&genetic.root[[q, j]]))
                        })
                    })
                    .collect()
            })
            .collect();
        let drive = penalty_ranges(&model.spec.drive_penalties, c, zero)?;
        for (index, penalty) in model.spec.drive_penalties.iter().enumerate() {
            let size = penalty.columns.len();
            let local = Arc::new(Array2::from_shape_fn((size * genes, size * genes), |(a, q)| {
                constant(penalty.local[[a / genes, q / genes]]).mul(&moment[a % genes][q % genes])
            }));
            let coordinates = (0..k)
                .map(|axis| {
                    let base = model.layout.drive.start + axis * c * genes;
                    base + penalty.columns.start * genes..base + penalty.columns.end * genes
                })
                .collect();
            self.push_gaussian(
                FunctionPenalty::DriveVariation { penalty: index },
                GaussianForm::Penalty {
                    local,
                    rank: penalty.rank * genes,
                },
                coordinates,
                None,
            );
        }
        for (range, half_log_det) in &drive {
            // det(sum_j S_j ⊗ M) = det(sum_j S_j)^genes det(M)^width.
            let half = half_log_det
                .scale(genes as f64)
                .add(&genetic.log_determinant.scale(0.5 * range.len() as f64));
            self.normalizer = self.normalizer.add(&half.scale(k as f64));
        }
        let mut entry_gram = vec![vec![zero.clone(); e]; e];
        let subject_weight = div(&constant(1.0), &constant(subjects));
        for h in histories {
            let entry: Vec<S> = std::iter::once(1.0)
                .chain(h.entry_design.iter().copied())
                .map(|v| constant(v))
                .collect();
            add_outer(&mut entry_gram, &entry, &subject_weight);
        }
        let entry = Arc::new(product_root(
            &FunctionRoot::from_gram(&entry_gram, zero)?,
            &genetic,
        ));
        // The per-signature entry block width, read from the law's layout rather than recounted.
        let entry_width = model.layout.entry.len() / k;
        self.normalizer = self
            .normalizer
            .add(&entry.log_determinant.scale(0.5 * k as f64));
        self.push_gaussian(
            FunctionPenalty::EntryMean,
            GaussianForm::Root(entry),
            (0..k)
                .map(|axis| {
                    let start = model.layout.entry.start + axis * entry_width;
                    start..start + e * genes
                })
                .collect(),
            None,
        );
        let mut prevalence = 0;
        for (mark, &kind) in model.spec.marks.iter().enumerate() {
            if kind == MarkKind::Once {
                self.normalizer = self
                    .normalizer
                    .add(&genetic.log_determinant.scale(0.5 * k as f64));
                self.push_gaussian(
                    FunctionPenalty::EntryPrevalence { mark },
                    GaussianForm::Root(Arc::clone(&genetic)),
                    (0..k)
                        .map(|axis| {
                            let start = model.layout.entry.start
                                + axis * entry_width
                                + (e + prevalence) * genes;
                            start..start + genes
                        })
                        .collect(),
                    None,
                );
                prevalence += 1;
            }
        }
        let unit = Arc::new(FunctionRoot::identity(k, zero));
        for (mark, jump) in model.layout.jumps.iter().enumerate() {
            if let Some(range) = jump {
                self.push_gaussian(
                    FunctionPenalty::DiseaseJump { mark },
                    GaussianForm::Root(Arc::clone(&unit)),
                    vec![range.clone()],
                    None,
                );
            }
        }
        // Loading levels: column 0 of each axis's loading function.
        let scalar = Arc::new(FunctionRoot::identity(1, zero));
        for (channel, family) in model.spec.measurements.iter().enumerate() {
            let start = model.layout.measurement_location[channel].start;
            let scale = matches!(family, MeasurementFamily::StudentT)
                .then_some(model.layout.measurement_shape[channel].start);
            self.push_gaussian(
                FunctionPenalty::MeasurementEffect { channel },
                GaussianForm::Root(Arc::clone(&scalar)),
                (0..k)
                    .map(|axis| {
                        let offset = start + (axis + 1) * p;
                        offset..offset + 1
                    })
                    .collect(),
                scale,
            );
        }
        Ok(())
    }

    pub fn penalties(&self) -> &[FunctionPenalty] {
        &self.penalties
    }
    /// The prior density and its derivatives at `theta` and one log strength per
    /// penalty. A Gaussian block whose scaled coefficients
    /// `exp(rho/2 - log scale) beta` overflow is refused as not representable:
    /// the strength limit is the restricted model, which the fit evaluates
    /// without forming such a precision.
    pub fn evaluate<'a>(
        &'a self,
        theta: &[S],
        log_strengths: &[S],
    ) -> Result<FunctionPriorEvaluation<'a, 'm, S>, EventHistoryError> {
        self.evaluate_excluding(theta, log_strengths, None)
    }

    /// The evaluation with one Gaussian family's blocks left out, for its zero-effect face.
    fn evaluate_excluding<'a>(
        &'a self,
        theta: &[S],
        log_strengths: &[S],
        excluded: Option<usize>,
    ) -> Result<FunctionPriorEvaluation<'a, 'm, S>, EventHistoryError> {
        self.model.validate_parameters(theta)?;
        if log_strengths.len() != self.penalties.len()
            || log_strengths.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "function prior requires one finite log strength per active function penalty",
            ));
        }
        let decoder = self
            .model
            .decoder_prior(theta, &log_strengths[..self.decoder_strengths])?;
        let mut gradient = decoder.gradient().to_vec();
        let category = self.category.evaluate(theta, &mut gradient);
        let zero = theta[0].constant_like(0.0);
        let mut strength_gradient = vec![zero.clone(); self.penalties.len()];
        let mut strength_second = vec![zero.clone(); self.penalties.len()];
        for d in 0..self.decoder_strengths {
            strength_gradient[d] = decoder.log_strength_gradient()[d].clone();
            strength_second[d] = decoder.log_strength_second_derivative()[d].clone();
        }
        let mut result = FunctionPriorEvaluation {
            prior: self,
            excluded,
            log_density: decoder
                .log_density()
                .add(category.log_density())
                .add(&self.normalizer),
            gradient,
            strength_gradient,
            strength_second,
            decoder,
            gaussian: Vec::with_capacity(self.gaussian.len()),
            structural: Vec::with_capacity(self.structural.len()),
            category,
            baseline_weights: Vec::with_capacity(self.model.spec.marks.len()),
        };
        for function in &self.gaussian {
            if excluded == Some(function.strength) {
                result.gaussian.push(GaussianEvaluation {
                    half_log_precision: zero.clone(),
                    weighted_penalty: zero.clone(),
                    gradients: Vec::new(),
                });
                continue;
            }
            let strength = &log_strengths[function.strength];
            let half = match function.log_scale {
                Some(j) => strength.scale(0.5).sub(&theta[j]),
                None => strength.scale(0.5),
            };
            let rank = (function.form.rank() * function.coordinates.len()) as f64;
            let mut energy = zero.clone();
            let mut gradients = Vec::with_capacity(function.coordinates.len());
            for range in &function.coordinates {
                let (block, positive) = function.form.quadratic(&theta[range.clone()], &half);
                energy = energy.add(&block);
                let gradient: Vec<S> = positive.iter().map(|v| v.neg()).collect();
                for (q, g) in range.clone().zip(&gradient) {
                    result.gradient[q] = result.gradient[q].add(g);
                }
                gradients.push(gradient);
            }
            result.log_density = result
                .log_density
                .add(&half.sub(&self.half_log_two_pi).scale(rank))
                .sub(&energy);
            let strength_score = add_real(&energy.neg(), 0.5 * rank);
            if let Some(q) = function.log_scale {
                result.gradient[q] = result.gradient[q].sub(&strength_score.scale(2.0));
            }
            let j = function.strength;
            result.strength_gradient[j] = result.strength_gradient[j].add(&strength_score);
            result.strength_second[j] = result.strength_second[j].sub(&energy);
            result.gaussian.push(GaussianEvaluation {
                half_log_precision: half,
                weighted_penalty: energy,
                gradients,
            });
        }
        // Independent exponential priors on the dimensionless geometric
        // baseline levels, with one shared learned strength across marks.
        // The change from intercept to level is triangular with the shape
        // coordinates and has Jacobian equal to that positive level.
        let rho = &log_strengths[self.level_strength];
        let design = &self.baseline_mean_design;
        let b = design.len();
        let marks = self.model.spec.marks.len();
        let mut baseline_weight = zero.clone();
        for mark in 0..marks {
            let start = self.model.layout.baseline.start + mark * b;
            let slopes = design[1..]
                .iter()
                .zip(&theta[start + 1..start + b])
                .fold(zero.clone(), |sum, (a, v)| sum.add(&a.mul(v)));
            let log_weight = rho
                .add(&theta[start])
                .add(&self.log_followup_scale)
                .add(&slopes);
            let weight = exp(&log_weight);
            result.log_density = result.log_density.add(&log_weight.sub(&weight));
            let complement = add_real(&weight.neg(), 1.0);
            for (j, a) in design.iter().enumerate() {
                result.gradient[start + j] = result.gradient[start + j].add(&complement.mul(a));
            }
            baseline_weight = baseline_weight.add(&weight);
            result.baseline_weights.push(weight);
        }
        result.strength_gradient[self.level_strength] =
            add_real(&baseline_weight.neg(), marks as f64);
        result.strength_second[self.level_strength] = baseline_weight.neg();
        for (functions, &index) in self.structural.iter().zip(&self.structural_strengths) {
            let rho = &log_strengths[index];
            let mut first = zero.clone();
            let mut second = zero.clone();
            let mut values = Vec::with_capacity(functions.len());
            for function in functions {
                let value = function.evaluate(theta, rho, &self.log_followup_scale);
                result.log_density = result.log_density.add(&value.log_density);
                result.gradient[value.coordinate] =
                    result.gradient[value.coordinate].add(&value.first);
                first = first.add(&value.strength_first);
                second = second.add(&value.strength_second);
                values.push(value);
            }
            result.strength_gradient[index] = first;
            result.strength_second[index] = second;
            result.structural.push(values);
        }
        if !result.log_density.value().is_finite()
            || result
                .gradient
                .iter()
                .chain(&result.strength_gradient)
                .chain(&result.strength_second)
                .any(|v| !v.value().is_finite())
        {
            return Err(numerical(
                "function prior or its derivatives are not representable",
            ));
        }
        Ok(result)
    }
}

impl<S: JetField> FunctionPriorEvaluation<'_, '_, S> {
    pub fn log_density(&self) -> &S {
        &self.log_density
    }
    pub fn gradient(&self) -> &[S] {
        &self.gradient
    }
    pub fn log_strength_gradient(&self) -> &[S] {
        &self.strength_gradient
    }
    /// Each strength enters one separable block of the density, so the
    /// strength Hessian conditional on coefficients is exactly this diagonal.
    /// Integrating coefficients introduces covariance terms; this is not REML
    /// curvature.
    pub fn log_strength_second_derivative(&self) -> &[S] {
        &self.strength_second
    }
    fn zero(&self) -> S {
        self.log_density.constant_like(0.0)
    }
    fn validate_direction(&self, direction: &[S]) -> Result<(), EventHistoryError> {
        if direction.len() != self.gradient.len() || direction.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "function prior direction has invalid dimensions or values",
            ));
        }
        Ok(())
    }
    /// One Gaussian function's `H v`, including its residual-scale couplings.
    /// Every term carries the block's precision exp(rho), so this is also the
    /// strength derivative of the function's `H v`.
    fn add_gaussian_hessian(&self, index: usize, direction: &[S], out: &mut [S]) {
        let function = &self.prior.gaussian[index];
        if self.excluded == Some(function.strength) {
            return;
        }
        let evaluation = &self.gaussian[index];
        let scale_direction = function.log_scale.map(|j| direction[j].clone());
        let mut cross = self.zero();
        for (range, gradient) in function.coordinates.iter().zip(&evaluation.gradients) {
            let (_, positive) = function
                .form
                .quadratic(&direction[range.clone()], &evaluation.half_log_precision);
            for ((q, g), h) in range.clone().zip(gradient).zip(&positive) {
                let mut value = out[q].add(h);
                if let Some(s) = &scale_direction {
                    value = value.add(&g.mul(s).scale(2.0));
                    cross = cross.add(&g.mul(&direction[q]).scale(2.0));
                }
                out[q] = value;
            }
        }
        if let (Some(q), Some(s)) = (function.log_scale, &scale_direction) {
            out[q] = out[q]
                .add(&cross)
                .add(&evaluation.weighted_penalty.mul(s).scale(4.0));
        }
    }
    /// The exponential level prior's `H v = w a (a . v)` per mark. It carries
    /// exp(rho) as a factor, so it is also its own strength derivative.
    fn add_level_hessian(&self, direction: &[S], out: &mut [S]) {
        let design = &self.prior.baseline_mean_design;
        let b = design.len();
        for (mark, weight) in self.baseline_weights.iter().enumerate() {
            let start = self.prior.model.layout.baseline.start + mark * b;
            let delta = design
                .iter()
                .zip(&direction[start..start + b])
                .fold(self.zero(), |sum, (a, v)| sum.add(&a.mul(v)));
            let product = weight.mul(&delta);
            for (j, a) in design.iter().enumerate() {
                out[start + j] = out[start + j].add(&product.mul(a));
            }
        }
    }
    /// `H v` with `H = -grad^2 log p` in coefficients.
    pub fn negative_hessian_product(&self, direction: &[S]) -> Result<Vec<S>, EventHistoryError> {
        // The decoder product validates the direction's length and values.
        let mut out = self.decoder.negative_hessian_product(direction)?;
        for index in 0..self.gaussian.len() {
            self.add_gaussian_hessian(index, direction, &mut out);
        }
        for values in &self.structural {
            for value in values {
                let c = value.coordinate;
                out[c] = out[c].sub(&value.second.mul(&direction[c]));
            }
        }
        self.category.add_negative_hessian(direction, &mut out);
        self.add_level_hessian(direction, &mut out);
        representable(out, "function prior Hessian product is not representable")
    }
    /// `d/d rho_j (H v)`, used for `tr(H^-1 d H/d rho_j)` in a Laplace evidence.
    pub fn strength_hessian_product(
        &self,
        strength: usize,
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        self.validate_direction(direction)?;
        if strength >= self.strength_gradient.len() {
            return Err(invalid(
                "function prior strength index is outside its penalties",
            ));
        }
        if strength < self.prior.decoder_strengths {
            return self.decoder.strength_hessian_product(strength, direction);
        }
        let mut out = vec![self.zero(); direction.len()];
        if strength == self.prior.level_strength {
            self.add_level_hessian(direction, &mut out);
        } else if let Some(position) = self
            .prior
            .structural_strengths
            .iter()
            .position(|&index| index == strength)
        {
            for value in &self.structural[position] {
                let c = value.coordinate;
                out[c] = out[c].sub(&value.strength_mixed_second.mul(&direction[c]));
            }
        } else {
            for index in 0..self.gaussian.len() {
                if self.prior.gaussian[index].strength == strength {
                    self.add_gaussian_hessian(index, direction, &mut out);
                }
            }
        }
        representable(
            out,
            "function prior strength curvature product is not representable",
        )
    }
    /// `D(H v)[delta]`: the coefficient third derivative of `-log p`,
    /// contracted with two directions.
    pub fn third_derivative_product(
        &self,
        delta: &[S],
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        // The decoder product validates both directions.
        let mut out = self.decoder.third_derivative_product(delta, direction)?;
        for (function, evaluation) in self.prior.gaussian.iter().zip(&self.gaussian) {
            // A fixed-precision Gaussian density is quadratic. Only a Student-t
            // residual scale, which multiplies the precision by exp(-2 q),
            // leaves third derivatives. With A the block precision and
            // G = A beta: dA/dq = -2A, dG/dq = -2G, and d(4E)/dq = -8E.
            let (Some(q), false) = (function.log_scale, self.excluded == Some(function.strength)) else {
                continue;
            };
            let (delta_q, direction_q) = (delta[q].clone(), direction[q].clone());
            let mut scale_total = evaluation
                .weighted_penalty
                .mul(&delta_q)
                .mul(&direction_q)
                .scale(-8.0);
            for (range, gradient) in function.coordinates.iter().zip(&evaluation.gradients) {
                let (_, precision_direction) = function
                    .form
                    .quadratic(&direction[range.clone()], &evaluation.half_log_precision);
                let (_, precision_delta) = function
                    .form
                    .quadratic(&delta[range.clone()], &evaluation.half_log_precision);
                // The stored gradient is -G.
                for (((i, g), av), ad) in range
                    .clone()
                    .zip(gradient)
                    .zip(&precision_direction)
                    .zip(&precision_delta)
                {
                    out[i] = out[i]
                        .sub(&delta_q.mul(av).scale(2.0))
                        .sub(&direction_q.mul(ad).scale(2.0))
                        .sub(&direction_q.mul(&delta_q).mul(g).scale(4.0));
                    scale_total = scale_total.sub(&direction[i].mul(ad).scale(2.0)).sub(
                        &g.mul(&delta_q.mul(&direction[i]).add(&direction_q.mul(&delta[i])))
                            .scale(4.0),
                    );
                }
            }
            out[q] = out[q].add(&scale_total);
        }
        for values in &self.structural {
            for value in values {
                let c = value.coordinate;
                out[c] = out[c].sub(&value.third.mul(&delta[c]).mul(&direction[c]));
            }
        }
        self.category.add_third_derivative(delta, direction, &mut out);
        let design = &self.prior.baseline_mean_design;
        let b = design.len();
        for (mark, weight) in self.baseline_weights.iter().enumerate() {
            let start = self.prior.model.layout.baseline.start + mark * b;
            let dot = |values: &[S]| {
                design
                    .iter()
                    .zip(&values[start..start + b])
                    .fold(self.zero(), |sum, (a, v)| sum.add(&a.mul(v)))
            };
            let product = weight.mul(&dot(direction)).mul(&dot(delta));
            for (j, a) in design.iter().enumerate() {
                out[start + j] = out[start + j].add(&product.mul(a));
            }
        }
        representable(
            out,
            "function prior third derivative product is not representable",
        )
    }
    /// Coefficient-by-strength mixed derivative product without storing a
    /// coefficients x strengths matrix.
    pub fn coefficient_strength_product(
        &self,
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        if direction.len() != self.strength_gradient.len()
            || direction.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "function prior strength direction has invalid dimensions or values",
            ));
        }
        let mut out = vec![self.zero(); self.gradient.len()];
        let layout = self.prior.model.layout();
        let k = self.prior.model.spec.signatures;
        let p = self.prior.model.spec.population_columns;
        for (d, value) in direction[..self.prior.decoder_strengths].iter().enumerate() {
            for axis in 0..k {
                out[layout.decoder.start + (d * k + axis) * p] =
                    self.decoder.coefficient_strength_cross()[d * k + axis].mul(value);
            }
        }
        for (function, evaluation) in self.prior.gaussian.iter().zip(&self.gaussian) {
            if self.excluded == Some(function.strength) {
                continue;
            }
            let value = &direction[function.strength];
            for (range, gradient) in function.coordinates.iter().zip(&evaluation.gradients) {
                for (q, g) in range.clone().zip(gradient) {
                    out[q] = out[q].add(&g.mul(value));
                }
            }
            if let Some(q) = function.log_scale {
                out[q] = out[q].add(&evaluation.weighted_penalty.mul(value).scale(2.0));
            }
        }
        for (values, &index) in self.structural.iter().zip(&self.prior.structural_strengths) {
            for value in values {
                let c = value.coordinate;
                out[c] = out[c].add(&value.mixed.mul(&direction[index]));
            }
        }
        let level_direction = &direction[self.prior.level_strength];
        let design = &self.prior.baseline_mean_design;
        let b = design.len();
        for (mark, weight) in self.baseline_weights.iter().enumerate() {
            let start = self.prior.model.layout.baseline.start + mark * b;
            for (j, a) in design.iter().enumerate() {
                out[start + j] = out[start + j].sub(&weight.mul(a).mul(level_direction));
            }
        }
        representable(
            out,
            "function prior mixed derivative product is not representable",
        )
    }
}

/// Coefficient coordinates of a zero-effect face, `theta = F gamma`. Each block maps its gamma
/// columns into one contiguous layout range through an orthonormal basis. A complete frame also
/// takes every coordinate outside its blocks as itself. Gamma follows ascending layout order,
/// with a block's columns at its range's start.
pub struct CoefficientFrame<S = f64> {
    width: usize,
    blocks: Vec<FrameBlock<S>>,
    complete: bool,
    zero: S,
}

/// One block of a frame: an orthonormal basis over one contiguous layout range.
pub struct FrameBlock<S = f64> {
    pub coordinates: Range<usize>,
    pub basis: Array2<S>,
}

impl<S: JetField> CoefficientFrame<S> {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn blocks(&self) -> &[FrameBlock<S>] {
        &self.blocks
    }
    /// The number of gamma coordinates.
    pub fn dimension(&self) -> usize {
        let columns: usize = self.blocks.iter().map(|block| block.basis.ncols()).sum();
        let covered: usize = self.blocks.iter().map(|block| block.coordinates.len()).sum();
        columns + if self.complete { self.width - covered } else { 0 }
    }
    /// `theta = F gamma`.
    pub fn embed(&self, gamma: &[S]) -> Vec<S> {
        let mut out = vec![self.zero.clone(); self.width];
        let mut next = 0;
        let mut column = 0;
        for block in &self.blocks {
            if self.complete {
                for q in next..block.coordinates.start {
                    out[q] = gamma[column].clone();
                    column += 1;
                }
            }
            for (i, q) in block.coordinates.clone().enumerate() {
                out[q] = (0..block.basis.ncols()).fold(self.zero.clone(), |sum, c| {
                    sum.add(&block.basis[[i, c]].mul(&gamma[column + c]))
                });
            }
            column += block.basis.ncols();
            next = block.coordinates.end;
        }
        if self.complete {
            for q in next..self.width {
                out[q] = gamma[column].clone();
                column += 1;
            }
        }
        out
    }
    /// `F' v` for a layout vector `v`.
    pub fn project(&self, vector: &[S]) -> Vec<S> {
        let mut out = Vec::with_capacity(self.dimension());
        let mut next = 0;
        for block in &self.blocks {
            if self.complete {
                out.extend_from_slice(&vector[next..block.coordinates.start]);
            }
            for c in 0..block.basis.ncols() {
                out.push(block.coordinates.clone().enumerate().fold(
                    self.zero.clone(),
                    |sum, (i, q)| sum.add(&block.basis[[i, c]].mul(&vector[q])),
                ));
            }
            next = block.coordinates.end;
        }
        if self.complete {
            out.extend_from_slice(&vector[next..]);
        }
        out
    }
    /// The dense `width x dimension` matrix `F`.
    pub fn dense(&self) -> Array2<S> {
        let dimension = self.dimension();
        let mut out = Array2::from_elem((self.width, dimension), self.zero.clone());
        for c in 0..dimension {
            let unit: Vec<S> = (0..dimension)
                .map(|q| self.zero.constant_like(f64::from(q == c)))
                .collect();
            for (q, value) in self.embed(&unit).into_iter().enumerate() {
                out[[q, c]] = value;
            }
        }
        out
    }
}

fn dot<S: JetField>(a: &[S], b: &[S], zero: &S) -> S {
    a.iter().zip(b).fold(zero.clone(), |sum, (x, y)| sum.add(&x.mul(y)))
}

/// `v` less its components along the orthonormal `basis`, by modified Gram-Schmidt applied
/// twice, which restores the orthogonality the first pass loses.
fn orthogonalized<S: JetField>(mut v: Vec<S>, basis: &[Vec<S>], zero: &S) -> Vec<S> {
    for _ in 0..2 {
        for u in basis {
            let component = dot(u, &v, zero);
            for (x, a) in v.iter_mut().zip(u) {
                *x = x.sub(&component.mul(a));
            }
        }
    }
    v
}

/// Extends the orthonormal `basis` by `count` of `candidates`, each time the one with the
/// largest residual. Selecting a declared count, rather than thresholding residuals, reads no
/// numerical rank.
fn extend_basis<S: JetField>(
    basis: &mut Vec<Vec<S>>,
    mut candidates: Vec<Vec<S>>,
    count: usize,
    zero: &S,
) -> Result<(), EventHistoryError> {
    for _ in 0..count {
        let residuals: Vec<Vec<S>> = candidates
            .iter()
            .map(|candidate| orthogonalized(candidate.clone(), basis, zero))
            .collect();
        let (index, square) = residuals
            .iter()
            .map(|residual| dot(residual, residual, zero))
            .enumerate()
            .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
            .ok_or_else(|| numerical("a face basis ran out of candidate directions"))?;
        if !(square.value().is_finite() && square.value() > 0.0) {
            return Err(numerical(
                "a penalty block's declared rank exceeds the range of its computed columns",
            ));
        }
        let length = sqrt(&square);
        basis.push(residuals[index].iter().map(|x| div(x, &length)).collect());
        candidates.swap_remove(index);
    }
    Ok(())
}

/// Orthonormal bases of the range of a symmetric positive-semidefinite block of declared rank,
/// from its own columns, and of that range's orthogonal complement, from unit vectors.
fn range_and_complement<S: JetField>(
    local: &Array2<S>,
    rank: usize,
    zero: &S,
) -> Result<(Array2<S>, Array2<S>), EventHistoryError> {
    let width = local.nrows();
    let mut basis = Vec::with_capacity(width);
    extend_basis(
        &mut basis,
        (0..width).map(|j| local.column(j).to_vec()).collect(),
        rank,
        zero,
    )?;
    let units = (0..width)
        .map(|j| (0..width).map(|i| zero.constant_like(f64::from(i == j))).collect())
        .collect();
    extend_basis(&mut basis, units, width - rank, zero)?;
    let columns = |range: Range<usize>| {
        Array2::from_shape_fn((width, range.len()), |(i, c)| basis[range.start + c][i].clone())
    };
    Ok((columns(0..rank), columns(rank..width)))
}

/// `B' A B`.
fn congruence<S: JetField>(basis: &Array2<S>, matrix: &Array2<S>, zero: &S) -> Array2<S> {
    let (n, m) = basis.dim();
    Array2::from_shape_fn((m, m), |(i, j)| {
        (0..n).fold(zero.clone(), |sum, a| {
            (0..n).fold(sum, |sum, b| {
                sum.add(&basis[[a, i]].mul(&matrix[[a, b]]).mul(&basis[[b, j]]))
            })
        })
    })
}

fn log_determinant_of<S: JetField>(matrix: &Array2<S>, zero: &S) -> Result<S, EventHistoryError> {
    let rows: Vec<Vec<S>> = matrix.rows().into_iter().map(|row| row.to_vec()).collect();
    Ok(FunctionRoot::from_gram(&rows, zero)?.log_determinant)
}

/// The zero-effect face of one Gaussian function family: the limit of its strength, where its
/// prior pins the family's penalized directions at zero. `pinned` spans those directions and
/// `penalty` is the family's precision on them at unit strength, block by block in `pinned`'s
/// order, in residual-scale units for the blocks `pinned_log_scales` names. `kept` spans every
/// other direction. Evaluations read the remaining families' prior over the kept coordinates,
/// with the restricted normalizer `½ log det(N' S_rest N)` of every penalty range the family
/// shares.
pub struct PriorFace<'a, 'm, S = f64> {
    prior: &'a JointFunctionPriors<'m, S>,
    strength: usize,
    pinned: CoefficientFrame<S>,
    penalty: Array2<S>,
    pinned_log_scales: Vec<Option<usize>>,
    kept: CoefficientFrame<S>,
    normalizer_change: S,
}

pub struct FaceEvaluation<'f, 'a, 'm, S = f64> {
    face: &'f PriorFace<'a, 'm, S>,
    full: FunctionPriorEvaluation<'a, 'm, S>,
    log_density: S,
    gradient: Vec<S>,
    strength_gradient: Vec<S>,
    strength_second: Vec<S>,
}

impl<'m, S: JetField> JointFunctionPriors<'m, S> {
    /// The zero-effect face of strength `strength`, or `None` when that strength's limit is not a
    /// coefficient subspace of the same law: decoder connections, the baseline level and the
    /// structural laws.
    pub fn zero_effect_face<'a>(
        &'a self,
        strength: usize,
    ) -> Result<Option<PriorFace<'a, 'm, S>>, EventHistoryError> {
        if strength >= self.penalties.len() {
            return Err(invalid(
                "function prior strength index is outside its penalties",
            ));
        }
        let zero = self.normalizer.constant_like(0.0);
        let mut pinned = Vec::new();
        let mut kept = Vec::new();
        let mut change = zero.clone();
        for function in self.gaussian.iter().filter(|f| f.strength == strength) {
            match &function.form {
                GaussianForm::Root(root) => {
                    let width = root.root.ncols();
                    let identity = Array2::from_shape_fn((width, width), |(i, j)| {
                        zero.constant_like(f64::from(i == j))
                    });
                    let gram = Array2::from_shape_fn((width, width), |(i, j)| {
                        (0..width).fold(zero.clone(), |sum, q| {
                            sum.add(&root.root[[q, i]].mul(&root.root[[q, j]]))
                        })
                    });
                    for range in &function.coordinates {
                        pinned.push((
                            FrameBlock {
                                coordinates: range.clone(),
                                basis: identity.clone(),
                            },
                            gram.clone(),
                            function.log_scale,
                        ));
                        kept.push(FrameBlock {
                            coordinates: range.clone(),
                            basis: Array2::from_elem((width, 0), zero.clone()),
                        });
                    }
                    change = change
                        .sub(&root.log_determinant.scale(0.5 * function.coordinates.len() as f64));
                }
                GaussianForm::Penalty { local, rank } => {
                    let (range_basis, complement) = range_and_complement(local, *rank, &zero)?;
                    let block_penalty = congruence(&range_basis, local, &zero);
                    for range in &function.coordinates {
                        // Every other family's blocks on this exact range.
                        let mut total: Array2<S> = (**local).clone();
                        let mut rest = Array2::from_elem(local.dim(), zero.clone());
                        for other in self.gaussian.iter().filter(|other| other.strength != strength) {
                            if let GaussianForm::Penalty { local: other_local, .. } = &other.form {
                                if other.coordinates.contains(range) {
                                    total.zip_mut_with(&**other_local, |t, o| *t = t.add(o));
                                    rest.zip_mut_with(&**other_local, |t, o| *t = t.add(o));
                                }
                            }
                        }
                        let restricted = congruence(&complement, &rest, &zero);
                        change = change.add(
                            &log_determinant_of(&restricted, &zero)?
                                .sub(&log_determinant_of(&total, &zero)?)
                                .scale(0.5),
                        );
                        pinned.push((
                            FrameBlock {
                                coordinates: range.clone(),
                                basis: range_basis.clone(),
                            },
                            block_penalty.clone(),
                            function.log_scale,
                        ));
                        kept.push(FrameBlock {
                            coordinates: range.clone(),
                            basis: complement.clone(),
                        });
                    }
                }
            }
        }
        if pinned.is_empty() {
            return Ok(None);
        }
        pinned.sort_by_key(|(block, _, _)| block.coordinates.start);
        kept.sort_by_key(|block| block.coordinates.start);
        let dimension: usize = pinned.iter().map(|(block, _, _)| block.basis.ncols()).sum();
        let mut penalty = Array2::from_elem((dimension, dimension), zero.clone());
        let mut offset = 0;
        for (_, block_penalty, _) in &pinned {
            let size = block_penalty.nrows();
            for i in 0..size {
                for j in 0..size {
                    penalty[[offset + i, offset + j]] = block_penalty[[i, j]].clone();
                }
            }
            offset += size;
        }
        let width = self.model.layout.width;
        let pinned_log_scales = pinned.iter().map(|(_, _, scale)| *scale).collect();
        Ok(Some(PriorFace {
            prior: self,
            strength,
            pinned: CoefficientFrame {
                width,
                blocks: pinned.into_iter().map(|(block, _, _)| block).collect(),
                complete: false,
                zero: zero.clone(),
            },
            penalty,
            pinned_log_scales,
            kept: CoefficientFrame {
                width,
                blocks: kept,
                complete: true,
                zero,
            },
            normalizer_change: change,
        }))
    }
}

impl<'a, 'm, S: JetField> PriorFace<'a, 'm, S> {
    pub fn pinned(&self) -> &CoefficientFrame<S> {
        &self.pinned
    }
    /// The family's precision on the pinned directions at unit strength.
    pub fn penalty(&self) -> &Array2<S> {
        &self.penalty
    }
    /// For each pinned block, the Student-t log-scale coordinate `q` whose `exp(-2 q)`
    /// multiplies that block's penalty.
    pub fn pinned_log_scales(&self) -> &[Option<usize>] {
        &self.pinned_log_scales
    }
    pub fn kept(&self) -> &CoefficientFrame<S> {
        &self.kept
    }
    /// The remaining strengths, in `JointFunctionPriors::penalties` order without this face's.
    pub fn penalties(&self) -> Vec<FunctionPenalty> {
        self.prior
            .penalties
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != self.strength)
            .map(|(_, label)| label.clone())
            .collect()
    }
    fn full_strengths(&self, reduced: &[S]) -> Vec<S> {
        let mut out = reduced.to_vec();
        out.insert(self.strength, self.normalizer_change.constant_like(0.0));
        out
    }
    /// The remaining families' prior at `theta = N gamma`, with one log strength per remaining
    /// penalty.
    pub fn evaluate<'f>(
        &'f self,
        gamma: &[S],
        log_strengths: &[S],
    ) -> Result<FaceEvaluation<'f, 'a, 'm, S>, EventHistoryError> {
        if gamma.len() != self.kept.dimension()
            || log_strengths.len() + 1 != self.prior.penalties.len()
        {
            return Err(invalid(
                "a face evaluation needs one coordinate per kept direction and one strength per remaining penalty",
            ));
        }
        let theta = self.kept.embed(gamma);
        let full = self.prior.evaluate_excluding(
            &theta,
            &self.full_strengths(log_strengths),
            Some(self.strength),
        )?;
        let reduced = |values: &[S]| {
            values
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != self.strength)
                .map(|(_, v)| v.clone())
                .collect::<Vec<_>>()
        };
        Ok(FaceEvaluation {
            face: self,
            log_density: full.log_density.add(&self.normalizer_change),
            gradient: self.kept.project(&full.gradient),
            strength_gradient: reduced(&full.strength_gradient),
            strength_second: reduced(&full.strength_second),
            full,
        })
    }
}

impl<'f, 'a, 'm, S: JetField> FaceEvaluation<'f, 'a, 'm, S> {
    pub fn log_density(&self) -> &S {
        &self.log_density
    }
    /// The remaining families' evaluation over full coefficient vectors at `theta = N gamma`,
    /// with this face's family left out. Its gradient and products act in layout coordinates,
    /// along the pinned directions as well as the kept ones, and its strengths follow
    /// `JointFunctionPriors::penalties` order with the face's strength inert. Its log density
    /// omits the restricted normalizer, which `log_density` includes.
    pub fn rest(&self) -> &FunctionPriorEvaluation<'a, 'm, S> {
        &self.full
    }
    pub fn gradient(&self) -> &[S] {
        &self.gradient
    }
    pub fn log_strength_gradient(&self) -> &[S] {
        &self.strength_gradient
    }
    pub fn log_strength_second_derivative(&self) -> &[S] {
        &self.strength_second
    }
    fn embed_direction(&self, direction: &[S]) -> Result<Vec<S>, EventHistoryError> {
        if direction.len() != self.gradient.len() || direction.iter().any(|v| !v.value().is_finite())
        {
            return Err(invalid(
                "face direction has invalid dimensions or values",
            ));
        }
        Ok(self.face.kept.embed(direction))
    }
    /// `N' H N v`.
    pub fn negative_hessian_product(&self, direction: &[S]) -> Result<Vec<S>, EventHistoryError> {
        let theta_direction = self.embed_direction(direction)?;
        Ok(self
            .face
            .kept
            .project(&self.full.negative_hessian_product(&theta_direction)?))
    }
    /// `N' d/d rho_j (H N v)`, with `strength` indexing the remaining penalties.
    pub fn strength_hessian_product(
        &self,
        strength: usize,
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        // An index past the remaining penalties maps past the full ones, which the full
        // evaluation refuses.
        let full_strength = if strength >= self.face.strength {
            strength + 1
        } else {
            strength
        };
        let theta_direction = self.embed_direction(direction)?;
        Ok(self.face.kept.project(
            &self
                .full
                .strength_hessian_product(full_strength, &theta_direction)?,
        ))
    }
    /// `N' D(H N v)[N delta]`.
    pub fn third_derivative_product(
        &self,
        delta: &[S],
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        let theta_delta = self.embed_direction(delta)?;
        let theta_direction = self.embed_direction(direction)?;
        Ok(self.face.kept.project(
            &self
                .full
                .third_derivative_product(&theta_delta, &theta_direction)?,
        ))
    }
    /// `N'` times the coefficient-by-strength mixed derivative along remaining strengths.
    pub fn coefficient_strength_product(
        &self,
        direction: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        if direction.len() != self.strength_gradient.len() {
            return Err(invalid(
                "face strength direction has invalid dimensions",
            ));
        }
        Ok(self.face.kept.project(
            &self
                .full
                .coefficient_strength_product(&self.face.full_strengths(direction))?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::function_prior_tests::{
        category_oracle, channel_agrees, compensated_sum, exact, rule, structural_oracle,
    };
    use super::super::law::{CompensatorPoint, JointSpecification, MeasurementRecord};
    use super::*;
    use crate::scalar::Rows;
    use crate::test_support::{Bound, agrees, cholesky_log_det};

    fn penalty(columns: Range<usize>, local: Array2<f64>, rank: usize) -> BasisPenalty {
        BasisPenalty {
            columns,
            local,
            rank,
        }
    }

    /// Complementary rank-one blocks over population columns 1..3: directions (1,1) and (1,-1).
    fn population_penalties() -> Vec<BasisPenalty> {
        vec![
            penalty(1..3, ndarray::array![[0.9, 0.9], [0.9, 0.9]], 1),
            penalty(1..3, ndarray::array![[0.5, -0.5], [-0.5, 0.5]], 1),
        ]
    }

    /// Each node's anchor, then one cell point at the node time carrying its gap.
    fn points(times: &[f64]) -> Vec<CompensatorPoint> {
        let mut out = vec![CompensatorPoint {
            node: 0,
            time: times[0],
            weight: 0.0,
        }];
        for node in 1..times.len() {
            out.push(CompensatorPoint {
                node,
                time: times[node],
                weight: 0.0,
            });
            out.push(CompensatorPoint {
                node,
                time: times[node],
                weight: times[node] - times[node - 1],
            });
        }
        out
    }

    fn fixture() -> (JointLikelihood, Vec<JointHistory>, Vec<f64>) {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 2,
            marks: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
            baseline_columns: 2,
            population_columns: 3,
            drive_columns: 2,
            entry_columns: 1,
            baseline_penalties: vec![penalty(1..2, ndarray::array![[1.3]], 1)],
            drive_penalties: vec![penalty(1..2, ndarray::array![[0.8]], 1)],
            population_penalties: population_penalties(),
            measurements: vec![
                MeasurementFamily::StudentT,
                MeasurementFamily::Probit { categories: 2 },
                MeasurementFamily::Probit { categories: 4 },
                MeasurementFamily::NegativeBinomial,
            ],
            genetic_mean: vec![0.7, -0.3],
            genetic_precision: ndarray::array![[2.0, 0.4], [0.4, 1.0]],
        })
        .unwrap();
        let times = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let rows = points(&times);
        let histories: Vec<_> = [-1.0, 0.5, 2.0]
            .iter()
            .map(|&context| JointHistory {
                times: times.clone(),
                points: rows.clone(),
                events: vec![vec![]; 5],
                initially_at_risk: vec![true; 3],
                baseline_design: Array2::from_shape_fn((rows.len(), 2), |(r, j)| {
                    rows[r].time.powi(j as i32)
                }),
                population_design: Array2::from_shape_fn((rows.len(), 3), |(r, j)| {
                    rows[r].time.powi(j as i32)
                }),
                drive_design: ndarray::array![[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5]],
                entry_design: vec![context],
                genetics: vec![Some(0.3), None],
                measurements: (0..4)
                    .map(|channel| MeasurementRecord {
                        node: 4,
                        channel,
                        value: Some([0.4, 1.0, 2.0, 3.0][channel]),
                        exposure: (channel == 3).then_some(1.0),
                        after_event: false,
                    })
                    .collect(),
            })
            .collect();
        let theta = (0..model.layout.width)
            .map(|j| 0.2 * (j as f64 + 0.3).cos())
            .collect();
        (model, histories, theta)
    }

    fn baseline_model(penalties: Vec<BasisPenalty>, columns: usize) -> JointLikelihood {
        JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: columns,
            population_columns: 1,
            drive_columns: 0,
            entry_columns: 0,
            baseline_penalties: penalties,
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap()
    }

    /// Two subjects with different spans and unequal cell weights. The baseline
    /// columns are powers of calendar time, so no column is centered.
    fn baseline_histories(columns: usize) -> Vec<JointHistory> {
        [vec![0.0_f64, 1.0, 3.0], vec![2.0_f64, 2.5, 4.0]]
            .into_iter()
            .map(|times| {
                let rows = points(&times);
                JointHistory {
                    baseline_design: Array2::from_shape_fn((rows.len(), columns), |(r, j)| {
                        rows[r].time.powi(j as i32)
                    }),
                    population_design: Array2::ones((rows.len(), 1)),
                    drive_design: Array2::zeros((times.len() - 1, 0)),
                    events: vec![vec![]; times.len()],
                    initially_at_risk: vec![true],
                    entry_design: vec![],
                    genetics: vec![],
                    measurements: vec![],
                    points: rows,
                    times,
                }
            })
            .collect()
    }

    fn bounded<'m>(model: &'m JointLikelihood, histories: &[JointHistory]) -> JointFunctionPriors<'m, Bound> {
        model
            .function_priors_over(&histories.iter().collect::<Vec<_>>(), &Bound::exact(0.0))
            .unwrap()
    }

    fn constant(value: f64) -> Bound {
        Bound::exact(0.0).constant_like(value)
    }

    fn unit(width: usize, index: usize) -> Vec<Bound> {
        (0..width).map(|q| Bound::exact(f64::from(q == index))).collect()
    }

    /// `R'R` over tracked entries; each entry sums in the same order as its mirror, so the
    /// values are exactly symmetric.
    fn gram(root: &Array2<Bound>) -> Array2<Bound> {
        let n = root.ncols();
        Array2::from_shape_fn((n, n), |(i, j)| {
            (0..root.nrows()).fold(Bound::exact(0.0), |sum, k| {
                sum.add(&root[[k, i]].mul(&root[[k, j]]))
            })
        })
    }

    /// `beta[start + columns]' S beta[start + columns]` for one frozen block.
    fn penalty_energy(beta: &[Bound], start: usize, block: &BasisPenalty) -> Bound {
        let mut total = Bound::exact(0.0);
        for (i, ci) in block.columns.clone().enumerate() {
            for (j, cj) in block.columns.clone().enumerate() {
                total = total.add(
                    &beta[start + ci]
                        .mul(&beta[start + cj])
                        .mul(&constant(block.local[[i, j]])),
                );
            }
        }
        total
    }

    /// One drive or entry coefficient block's function value `a + b'g` at `gene = (1, g)`.
    fn genetic_function(beta: &[Bound], start: usize, gene: &[Bound]) -> Bound {
        gene.iter()
            .enumerate()
            .fold(Bound::exact(0.0), |sum, (g, x)| sum.add(&beta[start + g].mul(x)))
    }

    /// The prior's log density over any jet, with one Gaussian family optionally left out.
    fn oracle<S: JetField, P: JetField>(
        prior: &JointFunctionPriors<'_, P>,
        theta: &[S],
        rho: &[S],
        excluded: Option<usize>,
    ) -> S {
        let k = prior.model.spec.signatures;
        let columns = prior.model.spec.population_columns;
        let mut value = add_real(&theta[0].constant_like(0.0), prior.normalizer.value());
        for (d, strength) in rho[..prior.decoder_strengths].iter().enumerate() {
            let lambda = exp(strength);
            let mut logits = vec![theta[0].constant_like(0.0)];
            for j in 0..k {
                let logit = &theta[prior.model.layout.decoder.start + (d * k + j) * columns];
                logits.push(logit.clone());
                value = value.add(&ln(&add_real(&lambda, (j + 1) as f64))).add(logit);
            }
            value = value.sub(&add_real(&lambda, (k + 1) as f64).mul(&log_sum_exp(&logits)));
        }
        for function in &prior.gaussian {
            if excluded == Some(function.strength) {
                continue;
            }
            let mut precision = rho[function.strength].clone();
            if let Some(q) = function.log_scale {
                precision = precision.sub(&theta[q].scale(2.0));
            }
            let rank = function.form.rank() * function.coordinates.len();
            value = value.add(&precision.scale(0.5 * rank as f64)).sub(
                &ln(&theta[0].constant_like(2.0 * std::f64::consts::PI)).scale(0.5 * rank as f64),
            );
            for range in &function.coordinates {
                let beta = &theta[range.clone()];
                let zero = theta[0].constant_like(0.0);
                let energy = match &function.form {
                    GaussianForm::Root(root) => {
                        root.root.rows().into_iter().fold(zero.clone(), |sum, row| {
                            let dot = beta
                                .iter()
                                .zip(row)
                                .fold(zero.clone(), |s, (x, r)| s.add(&x.scale(r.value())));
                            sum.add(&dot.mul(&dot))
                        })
                    }
                    GaussianForm::Penalty { local, .. } => {
                        let n = beta.len();
                        (0..n).fold(zero.clone(), |sum, i| {
                            (0..n).fold(sum, |s, j| {
                                s.add(&beta[i].mul(&beta[j]).scale(local[[i, j]].value()))
                            })
                        })
                    }
                };
                value = value.sub(&exp(&precision).mul(&energy).scale(0.5));
            }
        }
        let b = prior.baseline_mean_design.len();
        for mark in 0..prior.model.spec.marks.len() {
            let start = prior.model.layout.baseline.start + mark * b;
            let log_level = theta[start..start + b]
                .iter()
                .zip(&prior.baseline_mean_design)
                .fold(
                    theta[0].constant_like(prior.log_followup_scale.value()),
                    |v, (x, a)| v.add(&x.scale(a.value())),
                );
            let total = rho[prior.level_strength].add(&log_level);
            value = value.add(&total).sub(&exp(&total));
        }
        value = value.add(&category_oracle(&prior.category, theta));
        for (functions, &index) in prior.structural.iter().zip(&prior.structural_strengths) {
            for function in functions {
                // The frozen log follow-up span is a shared input; production carries its rounding.
                value = value.add(&structural_oracle(
                    function,
                    theta,
                    &rho[index],
                    &theta[0].constant_like(prior.log_followup_scale.value()),
                ));
            }
        }
        value
    }

    /// Seeds under three nested directions: `inner` innermost, then `first`, then `second`.
    fn nested(
        values: &[f64],
        inner: &dyn Fn(usize) -> f64,
        first: &dyn Fn(usize) -> f64,
        second: &dyn Fn(usize) -> f64,
    ) -> Vec<Rows<Rows<Rows<Bound, 1>, 1>, 1>> {
        values
            .iter()
            .enumerate()
            .map(|(q, &v)| {
                Rows::seed(
                    Rows::seed(Rows::seed(Bound::exact(v), [inner(q)]), [first(q)]),
                    [second(q)],
                )
            })
            .collect()
    }

    #[test]
    fn positive_baseline_level_prior_has_the_poisson_gamma_limit_even_without_events() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Recurrent, MarkKind::Recurrent],
            baseline_columns: 1,
            population_columns: 1,
            drive_columns: 0,
            entry_columns: 0,
            baseline_penalties: vec![],
            drive_penalties: vec![],
            population_penalties: vec![],
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let times = vec![0.0, 1.0, 2.0];
        let rows = points(&times);
        let history = JointHistory {
            baseline_design: Array2::ones((rows.len(), 1)),
            population_design: Array2::ones((rows.len(), 1)),
            drive_design: Array2::zeros((2, 0)),
            events: vec![vec![]; 3],
            initially_at_risk: vec![true; 2],
            entry_design: vec![],
            genetics: vec![],
            measurements: vec![],
            points: rows,
            times,
        };
        let prior = bounded(&model, std::slice::from_ref(&history));
        assert_eq!(prior.penalties(), &[FunctionPenalty::BaselineLevel]);
        let rho = Bound::exact(0.3);
        let exposure = 3.0_f64;
        // Points span z = rate * hazard over [0, range].
        let range = 100.0_f64;
        for events in [0_i32, 5] {
            // The hazard posterior is Gamma(events+1, exposure + T exp(rho)), with span T = 2.
            let rate = add_real(&exp(&rho).scale(2.0), exposure);
            let posterior_mean = |order: usize| {
                let r = rule(order, 0.0, range);
                let mut mass = Vec::new();
                let mut mean = Vec::new();
                let mut errors = [0.0_f64; 2];
                for (point, weight) in r.points.iter().zip(&r.weights) {
                    let hazard = div(point, &rate);
                    let q = ln(&hazard);
                    let out = prior
                        .evaluate(&[Bound::exact(q.value), Bound::exact(-0.4)], &[rho])
                        .unwrap();
                    let w = exp(&out
                        .log_density()
                        .add(&q.scale(f64::from(events)))
                        .sub(&hazard.scale(exposure))
                        .sub(&q))
                    .mul(&div(weight, &rate));
                    // A node's relative error: its own account, its weight's certified error,
                    // and the log hazard's movement, by q's rounding and by the point's
                    // displacement relative to z, through |d log(integrand)/d log hazard|, below
                    // events + 1 + range for the mass and one more for the first moment.
                    let relative = w.rounding() / w.value
                        + r.weight_relative_error
                        + (f64::from(events) + 2.0 + range) * (q.rounding() + r.point_error / point.value);
                    let moment = w.mul(&hazard);
                    mass.push(w.value);
                    mean.push(moment.value);
                    errors[0] += w.value * relative;
                    errors[1] += moment.value * relative + moment.rounding();
                }
                let (m, m_summation) = compensated_sum(&mass);
                let (h, h_summation) = compensated_sum(&mean);
                let ratio = h / m;
                (
                    ratio,
                    ratio
                        * ((errors[1] + h_summation) / h
                            + (errors[0] + m_summation) / m
                            + 2.0 * f64::EPSILON),
                )
            };
            let (coarse, _) = posterior_mean(129);
            let (fine, rounding) = posterior_mean(257);
            // Gamma(a, range)/Gamma(a) = e^-range sum_{k<a} range^k/k! for integer a: the
            // truncated mass (a = events+1) and first moment (a = events+2), relative.
            let upper = |a: i32| {
                (-range).exp()
                    * (0..a)
                        .map(|k| range.powi(k) / (1..=k).map(f64::from).product::<f64>())
                        .sum::<f64>()
            };
            let expected = div(&constant(f64::from(events + 1)), &rate);
            let bar = (fine - coarse).abs()
                + expected.value * (upper(events + 1) + upper(events + 2))
                + rounding
                + expected.rounding();
            assert!(
                (fine - expected.value).abs() <= bar,
                "events {events}: {fine} vs {} within {bar}",
                expected.value
            );
        }
        // theta0 = -1e200 and rho = 1e200 cancel exactly, so each mark's level weight is
        // exp(log T) with T = 2.
        let extreme = prior
            .evaluate(&exact(&[-1e200, -1e200]), &exact(&[1e200]))
            .unwrap();
        let weight = exp(&prior.log_followup_scale);
        for g in extreme.gradient() {
            agrees(g, &add_real(&weight.neg(), 1.0), "extreme gradient");
        }
        agrees(
            &extreme.log_strength_gradient()[0],
            &add_real(&weight.scale(2.0).neg(), 2.0),
            "extreme strength score",
        );
    }

    #[test]
    fn baseline_variation_and_level_priors_normalize_over_their_coefficients() {
        let model = baseline_model(vec![penalty(1..2, ndarray::array![[1.7]], 1)], 2);
        let histories = baseline_histories(2);
        let prior = bounded(&model, &histories);
        assert_eq!(
            prior.penalties(),
            &[
                FunctionPenalty::BaselineVariation { penalty: 0 },
                FunctionPenalty::BaselineLevel
            ]
        );
        let rho = [Bound::exact(0.3), Bound::exact(-0.2)];
        // The variation precision exp(rho_0) S with the frozen 1x1 block.
        let precision = exp(&rho[0]).mul(&constant(1.7));
        // Slopes over +-range standard deviations, with edges formed over Bound for the
        // truncation term; levels s in [low, high].
        let range = 14.0_f64;
        let slope_sd = div(&Bound::exact(1.0), &sqrt(&precision));
        let (slope_low, slope_high) = (slope_sd.scale(-range), slope_sd.scale(range));
        let (low, high) = (-40.0_f64, 4.0_f64);
        let a = prior.baseline_mean_design[1].value.abs();
        let integrate = |slope_order: usize, level_order: usize| {
            let slopes = rule(slope_order, slope_low.value, slope_high.value);
            let levels = rule(level_order, low, high);
            let mut mass = Vec::new();
            let mut second = Vec::new();
            let mut errors = [0.0_f64; 2];
            for (slope, ws) in slopes.points.iter().zip(&slopes.weights) {
                let shift = rho[1]
                    .add(&prior.log_followup_scale)
                    .add(&prior.baseline_mean_design[1].mul(slope));
                for (s, wl) in levels.points.iter().zip(&levels.weights) {
                    let theta = [s.sub(&shift), *slope];
                    let out = prior
                        .evaluate(&exact(&[theta[0].value, theta[1].value]), &rho)
                        .unwrap();
                    let density = exp(out.log_density()).mul(ws).mul(wl);
                    let g = [out.gradient()[0].value.abs(), out.gradient()[1].value.abs()];
                    // Point displacements move the evaluation point along the chart: a level point
                    // moves theta_0, a slope point moves theta_1 and theta_0 through the mean design.
                    let displacement = (g[1] + a * g[0]) * slopes.point_error + g[0] * levels.point_error;
                    // A node's relative error: its own account, both weights' certified errors,
                    // the displacement, and the evaluation point's rounding through the gradient.
                    let relative = density.rounding() / density.value
                        + slopes.weight_relative_error
                        + levels.weight_relative_error
                        + displacement
                        + (0..2).map(|k| g[k] * theta[k].rounding()).sum::<f64>();
                    let moment = density.mul(slope).mul(slope);
                    mass.push(density.value);
                    second.push(moment.value);
                    errors[0] += density.value * relative;
                    // The slope moment itself moves by 2|slope| per unit slope displacement.
                    errors[1] += moment.value * relative
                        + moment.rounding()
                        + density.value * 2.0 * slope.value.abs() * slopes.point_error;
                }
            }
            let (m, m_summation) = compensated_sum(&mass);
            let (v, v_summation) = compensated_sum(&second);
            [(m, errors[0] + m_summation), (v, errors[1] + v_summation)]
        };
        let coarse = integrate(129, 257);
        let fine = integrate(257, 513);
        // Truncation, never exact: the slope marginal's mass and second moment beyond either
        // edge by Mills' ratio at the edges' true distances, and the level mass outside
        // [low, high], below exp(low) + exp(-exp(high)).
        let gaussian_tail = truncated_mass(range, &slope_low, &slope_sd)
            + truncated_mass(range, &slope_high, &slope_sd);
        let second_tail = truncated_second_moment(range, &slope_low, &slope_sd)
            + truncated_second_moment(range, &slope_high, &slope_sd);
        let level_tail = (1.0 + 8.0 * f64::EPSILON) * (low.exp() + (-high.exp()).exp());
        let bar = (fine[0].0 - coarse[0].0).abs() + gaussian_tail + level_tail + fine[0].1;
        assert!((fine[0].0 - 1.0).abs() <= bar, "mass {} within {bar}", fine[0].0);
        let exact_second = div(&Bound::exact(1.0), &precision);
        let bar = (fine[1].0 - coarse[1].0).abs()
            + exact_second.value * (second_tail + level_tail)
            + fine[1].1
            + exact_second.rounding();
        assert!(
            (fine[1].0 - exact_second.value).abs() <= bar,
            "second moment {} vs {} within {bar}",
            fine[1].0,
            exact_second.value
        );
    }

    #[test]
    fn roots_and_penalty_ranges_carry_exact_log_determinants() {
        let (model, histories, _) = fixture();
        let prior = bounded(&model, &histories);
        let mut roots = 0;
        for (index, function) in prior.gaussian.iter().enumerate() {
            if let GaussianForm::Root(root) = &function.form {
                // The shared log-det oracle; its bar rests on the cited runtime-libm ln charge.
                let direct = cholesky_log_det(&gram(&root.root)).unwrap();
                channel_agrees(&root.log_determinant, &direct, &format!("root log determinant {index}"));
                roots += 1;
            }
        }
        // The entry mean, one prevalence, two jumps and four loading levels.
        assert_eq!(roots, 8);
        // The double penalty's separable determinant: det(l1 S1 + l2 S2) = l1^r1 l2^r2 det(S1+S2).
        let blocks = population_penalties();
        let zero = Bound::exact(0.0);
        let ranges = penalty_ranges(&blocks, 3, &zero).unwrap();
        assert_eq!(ranges.len(), 1);
        for (rho1, rho2) in [(-2.0, 3.0), (0.4, 0.4), (7.0, -5.0)] {
            let lambda = [exp(&Bound::exact(rho1)), exp(&Bound::exact(rho2))];
            let precision = Array2::from_shape_fn((2, 2), |(i, j)| {
                lambda[0]
                    .mul(&constant(blocks[0].local[[i, j]]))
                    .add(&lambda[1].mul(&constant(blocks[1].local[[i, j]])))
            });
            let direct = cholesky_log_det(&precision).unwrap();
            let separable = ranges[0]
                .1
                .scale(2.0)
                .add(&Bound::exact(rho1))
                .add(&Bound::exact(rho2));
            agrees(&separable, &direct, &format!("separable determinant at {rho1}, {rho2}"));
        }
        // Refusals, each by its own check: a positive-definite block whose declared rank does not
        // fill its range, an unpenalized non-constant column, and a singular sum. A constant basis
        // has no ranges.
        assert!(penalty_ranges(&[penalty(1..3, ndarray::array![[1.1, 0.3], [0.3, 0.7]], 1)], 3, &zero).is_err());
        assert!(penalty_ranges(&[penalty(1..2, ndarray::array![[1.0]], 1)], 3, &zero).is_err());
        let singular = vec![
            penalty(1..3, ndarray::array![[0.9, 0.9], [0.9, 0.9]], 1),
            penalty(1..3, ndarray::array![[1.8, 1.8], [1.8, 1.8]], 1),
        ];
        assert!(penalty_ranges(&singular, 3, &zero).is_err());
        assert!(penalty_ranges(&[], 1, &zero).unwrap().is_empty());
        // E[(a+b'g)^2] under g ~ N(mu, Sigma) is (a + b'mu)^2 + b' Sigma b: the DriveLevel and
        // GeneticDrive blocks, of ranks one and two, whose sum is the Gram
        // [[1, mu'], [mu, mu mu' + Sigma]] with determinant det(Sigma) = 1/det(precision).
        let local_of = |label: FunctionPenalty| {
            prior
                .gaussian
                .iter()
                .find(|f| prior.penalties[f.strength] == label)
                .map(|f| match &f.form {
                    GaussianForm::Penalty { local, rank } => (Arc::clone(local), *rank),
                    GaussianForm::Root(_) => unreachable!("the drive-level blocks are penalties"),
                })
                .unwrap()
        };
        let (level, level_rank) = local_of(FunctionPenalty::DriveLevel);
        let (dependence, dependence_rank) = local_of(FunctionPenalty::GeneticDrive);
        assert_eq!((level_rank, dependence_rank), (1, 2));
        let precision_det = Bound::exact(2.0).sub(&Bound::exact(0.4).mul(&Bound::exact(0.4)));
        let sigma = [
            [div(&Bound::exact(1.0), &precision_det), div(&Bound::exact(-0.4), &precision_det)],
            [div(&Bound::exact(-0.4), &precision_det), div(&Bound::exact(2.0), &precision_det)],
        ];
        let mu = [Bound::exact(0.7), Bound::exact(-0.3)];
        let product = Array2::from_shape_fn((3, 3), |(i, j)| level[[i, j]].add(&dependence[[i, j]]));
        for i in 0..3 {
            for j in 0..3 {
                let expected = match (i, j) {
                    (0, 0) => Bound::exact(1.0),
                    (0, j) => mu[j - 1],
                    (i, 0) => mu[i - 1],
                    (i, j) => mu[i - 1].mul(&mu[j - 1]).add(&sigma[i - 1][j - 1]),
                };
                agrees(&product[[i, j]], &expected, &format!("genetic Gram {i},{j}"));
            }
        }
    }

    #[test]
    fn no_signal_evidence_is_maximized_at_the_constant_function_limit() {
        // A quadratic baseline surface with a positive-definite frozen penalty and a
        // Gaussian observation factor exp(-beta' P beta / 2) centered at zero: the data carry
        // no signal. The exact evidence integrates the prior's own normalized density, so its
        // strength score is E_post[conditional prior score]. It must be positive at every
        // finite strength and vanish in the constant limit.
        let local = ndarray::array![[1.1, 0.3], [0.3, 0.7]];
        let model = baseline_model(vec![penalty(1..3, local.clone(), 2)], 3);
        let histories = baseline_histories(3);
        let prior = bounded(&model, &histories);
        let p = [[2.0_f64, 0.3], [0.3, 1.0]];
        let penalty_matrix: Vec<Vec<Bound>> = (0..2)
            .map(|i| (0..2).map(|j| constant(local[[i, j]])).collect())
            .collect();
        // Three-point Gauss-Hermite, exact for the quadratic score in beta, its nodes and weights
        // formed over Bound.
        let root3 = sqrt(&constant(3.0));
        let nodes = [root3.neg(), Bound::exact(0.0), root3];
        let weights = [
            div(&constant(1.0), &constant(6.0)),
            div(&constant(2.0), &constant(3.0)),
            div(&constant(1.0), &constant(6.0)),
        ];
        let inverse_of = |a: &[Vec<Bound>]| {
            let det = a[0][0].mul(&a[1][1]).sub(&a[0][1].mul(&a[1][0]));
            (
                [
                    [div(&a[1][1], &det), div(&a[0][1].neg(), &det)],
                    [div(&a[1][0].neg(), &det), div(&a[0][0], &det)],
                ],
                det,
            )
        };
        let half_trace = |inverse: &[[Bound; 2]; 2]| {
            let mut total = Bound::exact(0.0);
            for i in 0..2 {
                for j in 0..2 {
                    total = total.add(&inverse[j][i].scale(p[i][j]));
                }
            }
            total.scale(0.5)
        };
        let (penalty_inverse, _) = inverse_of(&penalty_matrix);
        let mut previous: Option<[f64; 2]> = None;
        // The conditional score q/2 - lambda beta' S beta/2 cancels two O(1) summands, so its
        // rounding floor is eps times their magnitude. Its sign is resolvable only while
        // tr(P A^-1)/2 exceeds that floor; the last point probes the limit, where the score
        // is bounded above by tr(P (lambda S)^-1)/2 = O(exp(-rho)) and its sign is not
        // resolvable by this route.
        for (rho, resolvable) in [
            (-6.0_f64, true),
            (-1.0, true),
            (0.0, true),
            (2.0, true),
            (8.0, true),
            (30.0, false),
        ] {
            let lambda = exp(&Bound::exact(rho));
            let a: Vec<Vec<Bound>> = (0..2)
                .map(|i| {
                    (0..2)
                        .map(|j| lambda.mul(&penalty_matrix[i][j]).add(&Bound::exact(p[i][j])))
                        .collect()
                })
                .collect();
            let (inverse, det) = inverse_of(&a);
            let expected = half_trace(&inverse);
            // A >= lambda S, so tr(P A^-1)/2 <= tr(P (lambda S)^-1)/2.
            let bound = div(&half_trace(&penalty_inverse), &lambda);
            // Exact posterior quadrature with A = L L', beta = L^-T z: the conditional score
            // is quadratic in beta.
            let l00 = sqrt(&a[0][0]);
            let l10 = div(&a[1][0], &l00);
            let l11 = sqrt(&a[1][1].sub(&l10.mul(&l10)));
            let mut terms = Vec::new();
            let mut error = expected.rounding();
            for (z0, w0) in nodes.iter().zip(&weights) {
                for (z1, w1) in nodes.iter().zip(&weights) {
                    let beta1 = div(z1, &l11);
                    let beta0 = div(&z0.sub(&l10.mul(&beta1)), &l00);
                    let out = prior
                        .evaluate(
                            &exact(&[0.0, beta0.value, beta1.value]),
                            &exact(&[rho, 0.0]),
                        )
                        .unwrap();
                    let g = out.log_strength_gradient()[0].mul(&w0.mul(w1));
                    // Node rounding: the weighted score's own account, and the node map's
                    // rounding through |d score/d beta| = |lambda S beta|.
                    let sensitivity: f64 = out.gaussian[0].gradients[0]
                        .iter()
                        .map(|v| v.value.abs())
                        .sum();
                    error += g.rounding()
                        + w0.value * w1.value * sensitivity * (beta0.rounding() + beta1.rounding());
                    terms.push(g.value);
                }
            }
            let (score, summation) = compensated_sum(&terms);
            let score_bar = error + summation;
            assert!(
                (score - expected.value).abs() <= score_bar,
                "rho {rho}: {score} vs {} within {score_bar}",
                expected.value
            );
            if resolvable {
                // With agreement within score_bar, this floor resolves the computed score as
                // positive.
                assert!(
                    expected.value > 2.0 * score_bar,
                    "rho {rho}: the score {} is below twice its bar {score_bar}",
                    expected.value
                );
            }
            // log Z = log p_variation(0) + log 2 pi - log det(A)/2, exact for a Gaussian block
            // once the level prior, which integrates to one over the intercept, is removed.
            // log det(I + X) lies in [0, tr X] for X = (lambda S)^-1 P, so log Z lies in
            // [-bound, 0].
            let at_zero = prior
                .evaluate(&exact(&[0.0, 0.0, 0.0]), &exact(&[rho, 0.0]))
                .unwrap();
            let level = at_zero.baseline_weights[0];
            let evidence = at_zero
                .log_density()
                .sub(&ln(&level).sub(&level))
                .add(&ln(&constant(2.0 * std::f64::consts::PI)))
                .sub(&ln(&det).scale(0.5));
            let evidence_bar = evidence.rounding();
            assert!(evidence.value <= evidence_bar, "rho {rho}: evidence {}", evidence.value);
            assert!(evidence.value >= -(bound.value + bound.rounding()) - evidence_bar);
            if let Some([previous_evidence, previous_evidence_bar]) = previous {
                assert!(evidence.value - evidence_bar > previous_evidence + previous_evidence_bar);
            }
            previous = Some([evidence.value, evidence_bar]);
        }
    }

    #[test]
    fn function_prior_all_coefficient_strength_and_scale_curvatures_match_the_density() {
        // Measured bar: the decoder, category and structural routes pass softplus and
        // log-softplus through compose_unary, whose one-ulp charge has no cited accuracy.
        let (model, histories, values) = fixture();
        let prior = bounded(&model, &histories);
        let theta = exact(&values);
        let rho_values: Vec<_> = (0..prior.penalties().len())
            .map(|j| 0.5 * (j as f64).sin())
            .collect();
        let rho = exact(&rho_values);
        let out = prior.evaluate(&theta, &rho).unwrap();
        let p = theta.len();
        let total = p + rho.len();
        let hessian: Vec<_> = (0..p)
            .map(|j| out.negative_hessian_product(&unit(p, j)).unwrap())
            .collect();
        let cross: Vec<_> = (0..rho.len())
            .map(|j| out.coefficient_strength_product(&unit(rho.len(), j)).unwrap())
            .collect();
        let all: Vec<f64> = values.iter().chain(&rho_values).copied().collect();
        let mut resolved = [0_usize; 2];
        for i in 0..total {
            for j in 0..=i {
                let seeds: Vec<Rows<Rows<Bound, 1>, 1>> = all
                    .iter()
                    .enumerate()
                    .map(|(q, &v)| {
                        Rows::seed(Rows::seed(Bound::exact(v), [f64::from(q == i)]), [f64::from(q == j)])
                    })
                    .collect();
                let jet = oracle(&prior, &seeds[..p], &seeds[p..], None);
                let gradient = if i < p {
                    out.gradient()[i]
                } else {
                    out.log_strength_gradient()[i - p]
                };
                let second = if i < p {
                    hessian[j][i].neg()
                } else if j < p {
                    cross[i - p][j]
                } else if i == j {
                    out.log_strength_second_derivative()[i - p]
                } else {
                    Bound::exact(0.0)
                };
                let name = format!("{i},{j}");
                // The oracle reads production's normalizer, roots, design and log span as constants,
                // so this value arm checks the density's dependence on coefficients and strengths,
                // not the normalizer; the normalization, root determinant, energy, face mass and
                // drive-level tests check those.
                agrees(out.log_density(), &jet.base.base, &name);
                resolved[0] += usize::from(channel_agrees(&gradient, &jet.base.rows[0], &name));
                resolved[1] += usize::from(channel_agrees(&second, &jet.rows[0].rows[0], &name));
            }
        }
        assert!(resolved.iter().all(|&count| count > 0), "resolvable entries {resolved:?}");
        // Positive control for that guard: a product with a zero direction, against a jet seeded
        // along nothing, has only structural zeros, so its count is zero.
        let zero_product = out
            .coefficient_strength_product(&vec![Bound::exact(0.0); rho.len()])
            .unwrap();
        let unseeded: Vec<Rows<Rows<Bound, 1>, 1>> = all
            .iter()
            .map(|&v| Rows::seed(Rows::seed(Bound::exact(v), [0.0]), [0.0]))
            .collect();
        let jet = oracle(&prior, &unseeded[..p], &unseeded[p..], None);
        let control = (0..p)
            .filter(|&q| channel_agrees(&zero_product[q], &jet.rows[0].rows[0], "unseeded control"))
            .count();
        assert_eq!(control, 0);
    }

    #[test]
    fn function_prior_third_derivatives_and_strength_curvature_products_match_nested_jets() {
        // Measured bar: the decoder, category and structural routes pass softplus and
        // log-softplus through compose_unary, whose one-ulp charge has no cited accuracy.
        let (model, histories, values) = fixture();
        let prior = bounded(&model, &histories);
        let h = prior.penalties().len();
        let rho_values: Vec<_> = (0..h).map(|j| 0.5 * (j as f64).sin()).collect();
        let out = prior.evaluate(&exact(&values), &exact(&rho_values)).unwrap();
        let p = values.len();
        let direction: Vec<_> = (0..p).map(|q| 0.3 * (1.7 * q as f64 + 0.2).cos()).collect();
        let delta: Vec<_> = (0..p).map(|q| 0.5 * (0.9 * q as f64 + 1.1).sin()).collect();
        let third = out
            .third_derivative_product(&exact(&delta), &exact(&direction))
            .unwrap();
        let constant_strengths = nested(&rho_values, &|_| 0.0, &|_| 0.0, &|_| 0.0);
        let mut resolved = 0;
        for i in 0..p {
            let jet = oracle(
                &prior,
                &nested(&values, &|q| f64::from(q == i), &|q| direction[q], &|q| delta[q]),
                &constant_strengths,
                None,
            );
            resolved += usize::from(channel_agrees(
                &third[i].neg(),
                &jet.rows[0].rows[0].rows[0],
                &format!("third derivative at {i}"),
            ));
        }
        assert!(resolved > 0);
        // Positive control for that guard: a zero delta leaves only structural zeros.
        let zero_delta = vec![0.0; p];
        let zero_third = out
            .third_derivative_product(&exact(&zero_delta), &exact(&direction))
            .unwrap();
        let control = (0..p)
            .filter(|&i| {
                let jet = oracle(
                    &prior,
                    &nested(&values, &|q| f64::from(q == i), &|q| direction[q], &|_| 0.0),
                    &constant_strengths,
                    None,
                );
                channel_agrees(&zero_third[i].neg(), &jet.rows[0].rows[0].rows[0], "zero-delta control")
            })
            .count();
        assert_eq!(control, 0);
        for j in 0..h {
            let product = out.strength_hessian_product(j, &exact(&direction)).unwrap();
            let strengths = nested(&rho_values, &|r| f64::from(r == j), &|_| 0.0, &|_| 0.0);
            let mut resolved = 0;
            for i in 0..p {
                let jet = oracle(
                    &prior,
                    &nested(&values, &|_| 0.0, &|q| direction[q], &|q| f64::from(q == i)),
                    &strengths,
                    None,
                );
                resolved += usize::from(channel_agrees(
                    &product[i].neg(),
                    &jet.rows[0].rows[0].rows[0],
                    &format!("strength {j} curvature at {i}"),
                ));
            }
            assert!(resolved > 0, "strength {j} has no resolvable curvature entry");
        }
        assert!(out.strength_hessian_product(h, &exact(&direction)).is_err());
        assert!(out
            .third_derivative_product(&exact(&delta[1..]), &exact(&direction))
            .is_err());
    }

    #[test]
    fn level_and_variation_energies_equal_direct_integration_of_the_model_functions() {
        let (model, histories, values) = fixture();
        let prior = bounded(&model, &histories);
        let rho = exact(&vec![0.2; prior.penalties().len()]);
        let beta = exact(&values);
        let evaluation = prior.evaluate(&beta, &rho).unwrap();
        let zero = Bound::exact(0.0);
        let root3 = sqrt(&constant(3.0));
        let nodes = [root3.neg(), zero, root3];
        let weights = [
            div(&constant(1.0), &constant(6.0)),
            div(&constant(2.0), &constant(3.0)),
            div(&constant(1.0), &constant(6.0)),
        ];
        // Invert the fixture's 2x2 precision independently of the stored genetic root.
        let det = Bound::exact(2.0).sub(&Bound::exact(0.4).mul(&Bound::exact(0.4)));
        let l00 = sqrt(&div(&Bound::exact(1.0), &det));
        let l10 = div(&div(&Bound::exact(-0.4), &det), &l00);
        let l11 = sqrt(&div(&Bound::exact(2.0), &det).sub(&l10.mul(&l10)));
        let (k, p, c, b, genes) = (2, 3, 2, 2, 3);
        let blocks = population_penalties();
        let log_scale = model.layout.measurement_shape[0].start;
        // Three-point Gauss-Hermite in two standard normal coordinates: exact for the
        // quadratic functions of (g, z) integrated here.
        let integrate = |f: &dyn Fn(&[Bound; 3], &[Bound; 2]) -> Bound| {
            let mut total = zero;
            for a in 0..3 {
                for bq in 0..3 {
                    let gene = [
                        Bound::exact(1.0),
                        Bound::exact(0.7).add(&l00.mul(&nodes[a])),
                        Bound::exact(-0.3)
                            .add(&l10.mul(&nodes[a]))
                            .add(&l11.mul(&nodes[bq])),
                    ];
                    total = total.add(&weights[a].mul(&weights[bq]).mul(&f(&gene, &[nodes[a], nodes[bq]])));
                }
            }
            total
        };
        let mut labels: Vec<FunctionPenalty> = prior
            .gaussian
            .iter()
            .map(|f| prior.penalties[f.strength].clone())
            .collect();
        labels.sort();
        labels.dedup();
        // Baseline variation, two decoder and two measurement variation blocks, the drive level,
        // genetic dependence and variation, entry mean, one prevalence, two jumps and four loading
        // levels.
        assert_eq!(labels.len(), 16);
        for label in labels {
            let production = prior
                .gaussian
                .iter()
                .zip(&evaluation.gaussian)
                .filter(|(f, _)| prior.penalties[f.strength] == label)
                .fold(zero, |sum, (_, e)| sum.add(&e.weighted_penalty));
            let energy = match &label {
                FunctionPenalty::BaselineVariation { penalty } => (0..3).fold(zero, |sum, mark| {
                    sum.add(&penalty_energy(
                        &beta,
                        model.layout.baseline.start + mark * b,
                        &model.spec.baseline_penalties[*penalty],
                    ))
                }),
                FunctionPenalty::DecoderVariation { penalty } => (0..3 * k).fold(zero, |sum, function| {
                    sum.add(&penalty_energy(
                        &beta,
                        model.layout.decoder.start + function * p,
                        &blocks[*penalty],
                    ))
                }),
                FunctionPenalty::MeasurementVariation { penalty } => (0..4).fold(zero, |sum, channel| {
                    // Student-t functions in residual-scale units.
                    let units = if channel == 0 {
                        exp(&beta[log_scale].scale(-2.0))
                    } else {
                        Bound::exact(1.0)
                    };
                    (0..=k).fold(sum, |sum, function| {
                        sum.add(
                            &penalty_energy(
                                &beta,
                                model.layout.measurement_location[channel].start + function * p,
                                &blocks[*penalty],
                            )
                            .mul(&units),
                        )
                    })
                }),
                FunctionPenalty::DiseaseJump { mark } => (0..k).fold(zero, |sum, axis| {
                    let jump = model.jump(&beta, &[*mark], axis);
                    sum.add(&jump.mul(&jump))
                }),
                FunctionPenalty::DriveLevel => (0..k).fold(zero, |sum, axis| {
                    // The level at the mean score, (1, mu).
                    let mean = [Bound::exact(1.0), Bound::exact(0.7), Bound::exact(-0.3)];
                    let level = genetic_function(&beta, model.layout.drive.start + axis * c * genes, &mean);
                    sum.add(&level.mul(&level))
                }),
                FunctionPenalty::GeneticDrive => integrate(&|gene, _| {
                    (0..k).fold(zero, |sum, axis| {
                        // The dependence on the centred scores, (0, g - mu).
                        let centred = [
                            Bound::exact(0.0),
                            gene[1].sub(&Bound::exact(0.7)),
                            gene[2].sub(&Bound::exact(-0.3)),
                        ];
                        let dependence =
                            genetic_function(&beta, model.layout.drive.start + axis * c * genes, &centred);
                        sum.add(&dependence.mul(&dependence))
                    })
                }),
                FunctionPenalty::DriveVariation { penalty } => {
                    let block = &model.spec.drive_penalties[*penalty];
                    integrate(&|gene, _| {
                        let mut total = zero;
                        for axis in 0..k {
                            let function = |column: usize| {
                                genetic_function(
                                    &beta,
                                    model.layout.drive.start + (axis * c + column) * genes,
                                    gene,
                                )
                            };
                            for (i, ci) in block.columns.clone().enumerate() {
                                for (j, cj) in block.columns.clone().enumerate() {
                                    total = total.add(
                                        &function(ci)
                                            .mul(&function(cj))
                                            .mul(&constant(block.local[[i, j]])),
                                    );
                                }
                            }
                        }
                        total
                    })
                }
                FunctionPenalty::EntryMean | FunctionPenalty::EntryPrevalence { .. } => integrate(&|gene, _| {
                    let mut total = zero;
                    for h in &histories {
                        for axis in 0..k {
                            let absent = model.mean(
                                &beta,
                                model.layout.entry.start,
                                &[1.0, h.entry_design[0], 0.0],
                                &gene[1..],
                                axis,
                            );
                            let effect = if matches!(label, FunctionPenalty::EntryMean) {
                                absent
                            } else {
                                model
                                    .mean(
                                        &beta,
                                        model.layout.entry.start,
                                        &[1.0, h.entry_design[0], 1.0],
                                        &gene[1..],
                                        axis,
                                    )
                                    .sub(&absent)
                            };
                            total = total.add(&div(&effect.mul(&effect), &constant(3.0)));
                        }
                    }
                    total
                }),
                FunctionPenalty::MeasurementEffect { channel } => integrate(&|_, state| {
                    let start = model.layout.measurement_location[*channel].start;
                    let mut value = beta[start + p]
                        .mul(&state[0])
                        .add(&beta[start + 2 * p].mul(&state[1]));
                    if *channel == 0 {
                        value = value.mul(&exp(&beta[log_scale].neg()));
                    }
                    value.mul(&value)
                }),
                other => unreachable!("{other:?} is not a Gaussian family"),
            };
            let expected = exp(&Bound::exact(0.2)).mul(&energy).scale(0.5);
            agrees(&production, &expected, &format!("{label:?}"));
        }
    }

    #[test]
    fn function_prior_respects_basis_changes_and_rejects_aliased_bases() {
        let model = baseline_model(vec![penalty(1..2, ndarray::array![[1.3]], 1)], 2);
        let histories = baseline_histories(2);
        let prior = bounded(&model, &histories);
        let rho = exact(&[0.3, -0.1]);
        let values = [0.4_f64, -0.7];
        let out = prior.evaluate(&exact(&values), &rho).unwrap();
        // Column 1 becomes 7t + 3. The final function is unchanged under beta_1' = beta_1/7 and
        // beta_0' = beta_0 - 3 beta_1', and the frozen penalty of that final function is 49 S.
        let changed_model = baseline_model(vec![penalty(1..2, ndarray::array![[1.3 * 49.0]], 1)], 2);
        let mut transformed = histories.clone();
        for h in &mut transformed {
            h.baseline_design
                .column_mut(1)
                .mapv_inplace(|v| 7.0 * v + 3.0);
        }
        let changed = bounded(&changed_model, &transformed);
        let slope = div(&Bound::exact(values[1]), &constant(7.0));
        let beta = [Bound::exact(values[0]).sub(&slope.scale(3.0)), slope];
        let new = changed
            .evaluate(&exact(&[beta[0].value, beta[1].value]), &rho)
            .unwrap();
        // The new coordinates' rounding moves each new quantity through its own gradient.
        let chart: f64 = beta
            .iter()
            .zip(new.gradient())
            .map(|(x, g)| g.value.abs() * x.rounding())
            .sum();
        // The coordinate change has Jacobian 1/7, so the densities differ by log 7.
        let excess = new
            .log_density()
            .sub(out.log_density())
            .sub(&ln(&Bound::exact(7.0)));
        assert!(
            excess.value.abs() <= excess.rounding() + chart,
            "log Jacobian excess {} within {}",
            excess.value,
            excess.rounding() + chart
        );
        let (model, histories, _) = fixture();
        let mut aliased = histories.clone();
        for h in &mut aliased {
            h.entry_design[0] = 1.0;
        }
        assert!(
            model
                .function_priors(&aliased.iter().collect::<Vec<_>>())
                .is_err()
        );
    }

    /// An upper bound on the Gaussian mass beyond one edge of a truncated domain, relative to the
    /// whole. `edge` was formed over `Bound` as a mean minus or plus `range` standard deviations,
    /// so the true edge lies at least `x = range - edge.rounding()/(sd - sd.rounding())` true
    /// standard deviations from the true mean, and Mills' ratio bounds the mass beyond `x` by
    /// phi(x)/x. The inflation covers the few correctly rounded operations and the cited exp
    /// that form the bound itself.
    fn truncated_mass(range: f64, edge: &Bound, sd: &Bound) -> f64 {
        let x = range - edge.rounding() / (sd.value - sd.rounding());
        (1.0 + 8.0 * f64::EPSILON) * (-0.5 * x * x).exp() / (x * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// As `truncated_mass`, for the second moment about the mean beyond that edge, relative to
    /// the variance: E[Z^2; Z > x] = x phi(x) + Phi(-x) <= phi(x) (x + 1/x).
    fn truncated_second_moment(range: f64, edge: &Bound, sd: &Bound) -> f64 {
        let x = range - edge.rounding() / (sd.value - sd.rounding());
        (1.0 + 8.0 * f64::EPSILON) * (-0.5 * x * x).exp() * (x + 1.0 / x)
            / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// The face's log density over gamma from the full oracle with the face's family left out,
    /// at `theta = N gamma` with the computed kept basis read as constants.
    fn face_oracle<S: JetField>(
        prior: &JointFunctionPriors<'_, Bound>,
        face: &PriorFace<'_, '_, Bound>,
        kept: &Array2<Bound>,
        gamma: &[S],
        rho: &[S],
    ) -> S {
        let zero = gamma[0].constant_like(0.0);
        let theta: Vec<S> = (0..kept.nrows())
            .map(|q| {
                (0..kept.ncols()).fold(zero.clone(), |sum, c| sum.add(&gamma[c].scale(kept[[q, c]].value)))
            })
            .collect();
        let mut full_rho = rho.to_vec();
        full_rho.insert(face.strength, zero);
        add_real(
            &oracle(prior, &theta, &full_rho, Some(face.strength)),
            face.normalizer_change.value,
        )
    }

    #[test]
    fn a_baseline_zero_effect_face_pins_one_block_and_normalizes_the_remaining_prior() {
        // Two complementary rank-one blocks over baseline columns 1..3, in directions (1,1) and
        // (1,-1). The face of the first pins (1,1)/sqrt 2; the remaining prior over
        // (theta_0, gamma) must integrate to one, which checks the restricted normalizer
        // ½ log det(N' S_1 N) - ½ log det(S_0 + S_1).
        let model = baseline_model(population_penalties(), 3);
        let histories = baseline_histories(3);
        let prior = bounded(&model, &histories);
        assert_eq!(
            prior.penalties(),
            &[
                FunctionPenalty::BaselineVariation { penalty: 0 },
                FunctionPenalty::BaselineVariation { penalty: 1 },
                FunctionPenalty::BaselineLevel
            ]
        );
        assert!(prior.zero_effect_face(2).unwrap().is_none());
        assert!(prior.zero_effect_face(3).is_err());
        let face = prior.zero_effect_face(0).unwrap().unwrap();
        assert_eq!(
            face.penalties(),
            vec![
                FunctionPenalty::BaselineVariation { penalty: 1 },
                FunctionPenalty::BaselineLevel
            ]
        );
        assert_eq!((face.pinned().dimension(), face.kept().dimension()), (1, 2));
        assert_eq!(face.pinned_log_scales(), &[None]);
        let (r, n) = (face.pinned().dense(), face.kept().dense());
        // In exact arithmetic twice-applied Gram-Schmidt returns a unit range direction orthogonal to
        // its complement; each computed entry lies within its own running bound of that exact value.
        let near = |value: Bound, exact: f64, name: &str| {
            assert!(
                (value.value - exact).abs() <= value.rounding(),
                "{name}: {} against {exact}, bound {}",
                value.value,
                value.rounding()
            );
        };
        let column = |m: &Array2<Bound>, c: usize| m.column(c).to_vec();
        near(dot(&column(&r, 0), &column(&r, 0), &Bound::exact(0.0)), 1.0, "R'R");
        for a in 0..2 {
            near(dot(&column(&r, 0), &column(&n, a), &Bound::exact(0.0)), 0.0, "R'N");
        }
        // The pinned penalty R' S_0 R on (1,1)/sqrt 2 is 1.8.
        agrees(&face.penalty()[[0, 0]], &constant(1.8), "pinned penalty");
        // The kept gamma is (theta_0, the (1,-1) direction's coordinate), whose precision is
        // exp(rho_1) N' S_1 N = exp(rho_1).
        let rho = [Bound::exact(0.3), Bound::exact(-0.2)];
        // gamma's precision exp(rho_1) N' S_1 N, from the computed kept direction.
        let second = population_penalties()[1].local.mapv(constant);
        let restricted = (0..2).fold(Bound::exact(0.0), |sum, i| {
            (0..2).fold(sum, |sum, j| {
                sum.add(&n[[i + 1, 1]].mul(&second[[i, j]]).mul(&n[[j + 1, 1]]))
            })
        });
        let range = 14.0_f64;
        let sd = div(&exp(&rho[0].scale(-0.5)), &sqrt(&restricted));
        let (coordinate_low, coordinate_high) = (sd.scale(-range), sd.scale(range));
        let (low, high) = (-40.0_f64, 4.0_f64);
        let design = [
            prior.baseline_mean_design[1].value,
            prior.baseline_mean_design[2].value,
        ];
        let tilt = (design[0] * n[[1, 1]].value + design[1] * n[[2, 1]].value).abs();
        let integrate = |order: usize, level_order: usize| {
            let coordinates = rule(order, coordinate_low.value, coordinate_high.value);
            let levels = rule(level_order, low, high);
            let mut mass = Vec::new();
            let mut error = 0.0;
            for (g, wg) in coordinates.points.iter().zip(&coordinates.weights) {
                let slopes = face.kept().embed(&[Bound::exact(0.0), *g]);
                let shift = rho[1]
                    .add(&prior.log_followup_scale)
                    .add(&prior.baseline_mean_design[1].mul(&slopes[1]))
                    .add(&prior.baseline_mean_design[2].mul(&slopes[2]));
                for (s, wl) in levels.points.iter().zip(&levels.weights) {
                    let level = s.sub(&shift);
                    let out = face
                        .evaluate(&exact(&[level.value, g.value]), &rho)
                        .unwrap();
                    let density = exp(out.log_density()).mul(wg).mul(wl);
                    let gradient = [out.gradient()[0].value.abs(), out.gradient()[1].value.abs()];
                    // Displacements move gamma along the chart: a level point moves theta_0, a
                    // coordinate point moves gamma and theta_0 through the mean design.
                    let relative = density.rounding() / density.value
                        + coordinates.weight_relative_error
                        + levels.weight_relative_error
                        + (gradient[1] + tilt * gradient[0]) * (coordinates.point_error + g.rounding())
                        + gradient[0] * (levels.point_error + level.rounding());
                    mass.push(density.value);
                    error += density.value * relative;
                }
            }
            let (total, summation) = compensated_sum(&mass);
            (total, error + summation)
        };
        let (coarse, _) = integrate(129, 257);
        let (fine, rounding) = integrate(257, 513);
        // Truncation, never exact: gamma's mass beyond either edge by Mills' ratio, and the level
        // mass outside [low, high].
        let tails = truncated_mass(range, &coordinate_low, &sd)
            + truncated_mass(range, &coordinate_high, &sd)
            + (1.0 + 8.0 * f64::EPSILON) * (low.exp() + (-high.exp()).exp());
        let bar = (fine - coarse).abs() + tails + rounding;
        assert!((fine - 1.0).abs() <= bar, "face mass {fine} within {bar}");
    }

    #[test]
    fn zero_effect_face_products_match_nested_jets_over_the_kept_coordinates() {
        // Measured bar: the decoder, category and structural routes pass softplus and
        // log-softplus through compose_unary, whose one-ulp charge has no cited accuracy.
        let (model, histories, values) = fixture();
        let prior = bounded(&model, &histories);
        let h = prior.penalties().len();
        for label in [
            FunctionPenalty::MeasurementVariation { penalty: 0 },
            FunctionPenalty::GeneticDrive,
        ] {
            let j = prior
                .penalties()
                .iter()
                .position(|existing| *existing == label)
                .unwrap();
            let face = prior.zero_effect_face(j).unwrap().unwrap();
            let kept = face.kept().dense();
            let g = kept.ncols();
            assert_eq!(g + face.pinned().dimension(), model.layout.width);
            let gamma: Vec<f64> = face
                .kept()
                .project(&exact(&values))
                .iter()
                .map(|v| v.value)
                .collect();
            let rho_values: Vec<f64> = (0..h)
                .filter(|&r| r != j)
                .map(|r| 0.5 * (r as f64).sin())
                .collect();
            let out = face.evaluate(&exact(&gamma), &exact(&rho_values)).unwrap();
            let hessian: Vec<_> = (0..g)
                .map(|c| out.negative_hessian_product(&unit(g, c)).unwrap())
                .collect();
            let cross: Vec<_> = (0..h - 1)
                .map(|r| out.coefficient_strength_product(&unit(h - 1, r)).unwrap())
                .collect();
            let mut resolved = [0_usize; 5];
            for i in 0..g {
                for c in 0..=i {
                    let seeds: Vec<Rows<Rows<Bound, 1>, 1>> = gamma
                        .iter()
                        .enumerate()
                        .map(|(q, &v)| {
                            Rows::seed(Rows::seed(Bound::exact(v), [f64::from(q == i)]), [f64::from(q == c)])
                        })
                        .collect();
                    let strengths: Vec<_> = rho_values
                        .iter()
                        .map(|&v| Rows::seed(Rows::seed(Bound::exact(v), [0.0]), [0.0]))
                        .collect();
                    let jet = face_oracle(&prior, &face, &kept, &seeds, &strengths);
                    let name = format!("{label:?} {i},{c}");
                    // A derivative check: the oracle reads the face's normalizer change and the kept
                    // basis as constants, and the face mass and drive-level tests check those.
                    agrees(out.log_density(), &jet.base.base, &name);
                    resolved[0] += usize::from(channel_agrees(&out.gradient()[i], &jet.base.rows[0], &name));
                    resolved[1] += usize::from(channel_agrees(&hessian[c][i].neg(), &jet.rows[0].rows[0], &name));
                }
                for r in 0..h - 1 {
                    let seeds: Vec<Rows<Rows<Bound, 1>, 1>> = gamma
                        .iter()
                        .enumerate()
                        .map(|(q, &v)| Rows::seed(Rows::seed(Bound::exact(v), [f64::from(q == i)]), [0.0]))
                        .collect();
                    let strengths: Vec<_> = rho_values
                        .iter()
                        .enumerate()
                        .map(|(s, &v)| Rows::seed(Rows::seed(Bound::exact(v), [0.0]), [f64::from(s == r)]))
                        .collect();
                    let jet = face_oracle(&prior, &face, &kept, &seeds, &strengths);
                    resolved[2] += usize::from(channel_agrees(
                        &cross[r][i],
                        &jet.rows[0].rows[0],
                        &format!("{label:?} cross {r},{i}"),
                    ));
                }
            }
            let direction: Vec<_> = (0..g).map(|q| 0.3 * (1.7 * q as f64 + 0.2).cos()).collect();
            let delta: Vec<_> = (0..g).map(|q| 0.5 * (0.9 * q as f64 + 1.1).sin()).collect();
            let third = out
                .third_derivative_product(&exact(&delta), &exact(&direction))
                .unwrap();
            let constant_strengths = nested(&rho_values, &|_| 0.0, &|_| 0.0, &|_| 0.0);
            for i in 0..g {
                let jet = face_oracle(
                    &prior,
                    &face,
                    &kept,
                    &nested(&gamma, &|q| f64::from(q == i), &|q| direction[q], &|q| delta[q]),
                    &constant_strengths,
                );
                resolved[3] += usize::from(channel_agrees(
                    &third[i].neg(),
                    &jet.rows[0].rows[0].rows[0],
                    &format!("{label:?} third {i}"),
                ));
            }
            for r in 0..h - 1 {
                let product = out.strength_hessian_product(r, &exact(&direction)).unwrap();
                let strengths = nested(&rho_values, &|s| f64::from(s == r), &|_| 0.0, &|_| 0.0);
                for i in 0..g {
                    let jet = face_oracle(
                        &prior,
                        &face,
                        &kept,
                        &nested(&gamma, &|_| 0.0, &|q| direction[q], &|q| f64::from(q == i)),
                        &strengths,
                    );
                    resolved[4] += usize::from(channel_agrees(
                        &product[i].neg(),
                        &jet.rows[0].rows[0].rows[0],
                        &format!("{label:?} strength {r} curvature at {i}"),
                    ));
                }
            }
            assert!(resolved.iter().all(|&count| count > 0), "{label:?}: resolvable {resolved:?}");
            // Positive control for that guard: a zero direction leaves only structural zeros.
            let zero_product = out
                .strength_hessian_product(0, &vec![Bound::exact(0.0); g])
                .unwrap();
            let control = (0..g)
                .filter(|&i| {
                    let jet = face_oracle(
                        &prior,
                        &face,
                        &kept,
                        &nested(&gamma, &|_| 0.0, &|_| 0.0, &|q| f64::from(q == i)),
                        &nested(&rho_values, &|s| f64::from(s == 0), &|_| 0.0, &|_| 0.0),
                    );
                    channel_agrees(&zero_product[i].neg(), &jet.rows[0].rows[0].rows[0], "zero-direction control")
                })
                .count();
            assert_eq!(control, 0);
            assert!(out.strength_hessian_product(h - 1, &exact(&direction)).is_err());
            if label == FunctionPenalty::GeneticDrive {
                // At theta = N gamma the genetic dependences b are zero, so the full density exceeds
                // the face's by the dependence block's log density at zero, rank/2 (rho - log 2 pi)
                // with rank = count scores = 4, less the normalizer change the face swaps in:
                // ½ log det(N' S_level N) - ½ log det M per axis, with N' S_level N = 1 and
                // det M = det(Sigma) = 1/det(precision).
                let theta: Vec<f64> = face
                    .kept()
                    .embed(&exact(&gamma))
                    .iter()
                    .map(|v| v.value)
                    .collect();
                let mut full_rho = exact(&rho_values);
                full_rho.insert(j, Bound::exact(0.7));
                let full = prior.evaluate(&exact(&theta), &full_rho).unwrap();
                let precision_det = Bound::exact(2.0).sub(&Bound::exact(0.4).mul(&Bound::exact(0.4)));
                let expected = Bound::exact(0.7)
                    .sub(&ln(&constant(2.0 * std::f64::consts::PI)))
                    .scale(2.0)
                    .sub(&ln(&precision_det));
                agrees(
                    &full.log_density().sub(out.log_density()),
                    &expected,
                    "drive-level block at zero",
                );
            }
        }
    }

    #[test]
    fn a_variation_zero_effect_bayes_factor_is_exact_under_a_flat_student_t_intercept() {
        // Measured bar: the tail-inflation law passes softplus and log-softplus through
        // compose_unary, whose one-ulp charge has no cited accuracy.
        // Structure s1 carries intercept variation over population column 1; s0 is its zero
        // effect. Both share one Student-t channel's intercept level a under the flat measure
        // da, the REML treatment of an unpenalized coefficient. With a Gaussian observation
        // factor at residual scale sigma = exp(q) for y_i = a + b x_i, integrating a gives
        // sigma sqrt(2 pi/n) exp(-(Syy - 2 b Sxy + b^2 Sxx)/(2 sigma^2)) in both structures, and
        // every other law is identical at identical coordinates. So
        // log BF = log(lambda/(lambda + h))/2 + t^2/(2(lambda + h)), with lambda = exp(rho - 2q) S,
        // h = Sxx/sigma^2 and t = Sxy/sigma^2, which the variation block's normalizer must
        // reproduce. The prior takes no measure input, so the measure's scale cancelling belongs
        // to the evidence route that integrates the intercept.
        let channel_model = |columns: usize, penalties: Vec<BasisPenalty>| {
            JointLikelihood::new(JointSpecification {
                signatures: 0,
                marks: vec![MarkKind::Recurrent],
                baseline_columns: 1,
                population_columns: columns,
                drive_columns: 0,
                entry_columns: 0,
                baseline_penalties: vec![],
                drive_penalties: vec![],
                population_penalties: penalties,
                measurements: vec![MeasurementFamily::StudentT],
                genetic_mean: vec![],
                genetic_precision: Array2::zeros((0, 0)),
            })
            .unwrap()
        };
        let full = channel_model(2, vec![penalty(1..2, ndarray::array![[1.3]], 1)]);
        let reduced = channel_model(1, vec![]);
        let x = [0.2_f64, 0.5, 1.0, 1.4];
        let y = [0.3_f64, 0.9, 1.2, 2.1];
        let history = |columns: usize| {
            let times = vec![0.0, 1.0];
            let rows = points(&times);
            JointHistory {
                baseline_design: Array2::ones((rows.len(), 1)),
                population_design: Array2::from_shape_fn((rows.len(), columns), |(r, j)| {
                    rows[r].time.powi(j as i32)
                }),
                drive_design: Array2::zeros((1, 0)),
                events: vec![vec![]; 2],
                initially_at_risk: vec![true],
                entry_design: vec![],
                genetics: vec![],
                measurements: y
                    .iter()
                    .map(|&value| MeasurementRecord {
                        node: 1,
                        channel: 0,
                        value: Some(value),
                        exposure: None,
                        after_event: false,
                    })
                    .collect(),
                points: rows,
                times,
            }
        };
        let full_histories = [history(2)];
        let reduced_histories = [history(1)];
        let full_prior = bounded(&full, &full_histories);
        let reduced_prior = bounded(&reduced, &reduced_histories);
        assert_eq!(
            full_prior.penalties(),
            &[
                FunctionPenalty::MeasurementVariation { penalty: 0 },
                FunctionPenalty::BaselineLevel,
                FunctionPenalty::MeasurementPrecision { channel: 0 },
                FunctionPenalty::TailVarianceInflation { channel: 0 },
            ]
        );
        let (level, log_scale, raw) = (-0.2_f64, -0.3_f64, 0.6_f64);
        let rho_values = [0.4_f64, 0.1, -0.5, 0.2];
        let sigma = exp(&Bound::exact(log_scale));
        let lambda = exp(&Bound::exact(rho_values[0]).sub(&Bound::exact(log_scale).scale(2.0)))
            .mul(&constant(1.3));
        // The closed form, from centred sums.
        let mean = |v: &[f64]| {
            v.iter()
                .fold(Bound::exact(0.0), |sum, &z| sum.add(&constant(z)))
                .scale(0.25)
        };
        let (x_bar, y_bar) = (mean(&x), mean(&y));
        let mut sxx = Bound::exact(0.0);
        let mut sxy = Bound::exact(0.0);
        for (&xi, &yi) in x.iter().zip(&y) {
            let (u, v) = (constant(xi).sub(&x_bar), constant(yi).sub(&y_bar));
            sxx = sxx.add(&u.mul(&u));
            sxy = sxy.add(&u.mul(&v));
        }
        let variance = sigma.mul(&sigma);
        let h = div(&sxx, &variance);
        let t = div(&sxy, &variance);
        let total = lambda.add(&h);
        let expected = ln(&div(&lambda, &total))
            .scale(0.5)
            .add(&div(&t.mul(&t), &total).scale(0.5));
        // Posterior boxes of +-range marginal standard deviations, formed over Bound so the
        // truncation term reads the edges' own rounding. The full posterior precision is
        // [[n, Sum x], [Sum x, Sum x^2 + lambda sigma^2]]/sigma^2; the reduced one is n/sigma^2.
        let s2 = sigma.value * sigma.value;
        let total_of = |terms: Vec<Bound>| terms.iter().fold(Bound::exact(0.0), |sum, t| sum.add(t));
        let sum_x = total_of(x.iter().map(|&v| constant(v)).collect());
        let sum_xx = total_of(x.iter().map(|&v| constant(v).mul(&constant(v))).collect());
        let sum_y = total_of(y.iter().map(|&v| constant(v)).collect());
        let sum_xy = total_of(
            x.iter()
                .zip(&y)
                .map(|(&u, &v)| constant(u).mul(&constant(v)))
                .collect(),
        );
        let p00 = div(&Bound::exact(4.0), &variance);
        let p01 = div(&sum_x, &variance);
        let p11 = div(&sum_xx, &variance).add(&lambda);
        let det = p00.mul(&p11).sub(&p01.mul(&p01));
        let right = [div(&sum_y, &variance), div(&sum_xy, &variance)];
        let center = [
            div(&p11.mul(&right[0]).sub(&p01.mul(&right[1])), &det),
            div(&p00.mul(&right[1]).sub(&p01.mul(&right[0])), &det),
        ];
        let sd = [sqrt(&div(&p11, &det)), sqrt(&div(&p00, &det))];
        let (reduced_center, reduced_sd) = (sum_y.scale(0.25), sigma.scale(0.5));
        let range = 12.0_f64;
        let edges = |mean: &Bound, deviation: &Bound| {
            let half = deviation.scale(range);
            (mean.sub(&half), mean.add(&half))
        };
        let (a_edges, b_edges) = (edges(&center[0], &sd[0]), edges(&center[1], &sd[1]));
        let reduced_edges = edges(&reduced_center, &reduced_sd);
        let log_likelihood = |a: &Bound, b: &Bound| {
            x.iter().zip(&y).fold(Bound::exact(0.0), |sum, (&xi, &yi)| {
                let residual = div(&constant(yi).sub(a).sub(&b.mul(&constant(xi))), &sigma);
                sum.sub(&residual.mul(&residual).scale(0.5))
            })
        };
        let score = |a: f64, b: f64| {
            x.iter().zip(&y).fold([0.0_f64, 0.0_f64], |acc, (&xi, &yi)| {
                let r = (yi - a - b * xi) / s2;
                [acc[0] + r, acc[1] + xi * r]
            })
        };
        // Each node's term without the measure scale, and its relative error: both weights'
        // certified errors, and each point's displacement and rounding through the integrand's
        // gradient. The term's own rounding is charged after scaling.
        let full_nodes = |order: usize| {
            let ra = rule(order, a_edges.0.value, a_edges.1.value);
            let rb = rule(order, b_edges.0.value, b_edges.1.value);
            let rho = exact(&rho_values);
            let mut nodes = Vec::with_capacity(order * order);
            for (a, wa) in ra.points.iter().zip(&ra.weights) {
                for (b, wb) in rb.points.iter().zip(&rb.weights) {
                    let out = full_prior
                        .evaluate(&exact(&[level, a.value, b.value, log_scale, raw]), &rho)
                        .unwrap();
                    let term = exp(&out.log_density().add(&log_likelihood(a, b)))
                        .mul(wa)
                        .mul(wb);
                    let [ga, gb] = score(a.value, b.value);
                    let slope = [
                        (ga + out.gradient()[1].value).abs(),
                        (gb + out.gradient()[2].value).abs(),
                    ];
                    let relative = ra.weight_relative_error
                        + rb.weight_relative_error
                        + slope[0] * (ra.point_error + a.rounding())
                        + slope[1] * (rb.point_error + b.rounding());
                    nodes.push((term, relative));
                }
            }
            nodes
        };
        let reduced_nodes = |order: usize| {
            let ra = rule(order, reduced_edges.0.value, reduced_edges.1.value);
            let rho = exact(&rho_values[1..]);
            let zero = Bound::exact(0.0);
            ra.points
                .iter()
                .zip(&ra.weights)
                .map(|(a, wa)| {
                    let out = reduced_prior
                        .evaluate(&exact(&[level, a.value, log_scale, raw]), &rho)
                        .unwrap();
                    let term = exp(&out.log_density().add(&log_likelihood(a, &zero))).mul(wa);
                    let slope = (score(a.value, 0.0)[0] + out.gradient()[1].value).abs();
                    (
                        term,
                        ra.weight_relative_error + slope * (ra.point_error + a.rounding()),
                    )
                })
                .collect::<Vec<_>>()
        };
        let evidence = |nodes: &[(Bound, f64)]| {
            let mut values = Vec::with_capacity(nodes.len());
            let mut error = 0.0;
            for (term, relative) in nodes {
                values.push(term.value);
                error += term.rounding() + term.value * relative;
            }
            let (sum, summation) = compensated_sum(&values);
            (sum, error + summation)
        };
        let nodes = [
            full_nodes(129),
            full_nodes(257),
            reduced_nodes(129),
            reduced_nodes(257),
        ];
        // Truncation, never exact. Both integrands are Gaussian in their coordinates, so the mass
        // outside a box is below the sum of each marginal's mass beyond each edge, bounded by
        // Mills' ratio at the edges' true distances: two marginals for s1 and one for s0.
        let tail = [
            (&a_edges.0, &sd[0]),
            (&a_edges.1, &sd[0]),
            (&b_edges.0, &sd[1]),
            (&b_edges.1, &sd[1]),
            (&reduced_edges.0, &reduced_sd),
            (&reduced_edges.1, &reduced_sd),
        ]
        .iter()
        .map(|(edge, deviation)| truncated_mass(range, edge, deviation))
        .sum::<f64>();
        let [full_coarse, full_fine, reduced_coarse, reduced_fine] =
            [&nodes[0], &nodes[1], &nodes[2], &nodes[3]].map(|n| evidence(n));
        let relative = ((full_fine.0 - full_coarse.0).abs() + full_fine.1) / full_fine.0
            + ((reduced_fine.0 - reduced_coarse.0).abs() + reduced_fine.1) / reduced_fine.0
            + tail;
        let (full_log, reduced_log) = (full_fine.0.ln(), reduced_fine.0.ln());
        let log_factor = full_log - reduced_log;
        let bar = relative
            + f64::EPSILON * (full_log.abs() + reduced_log.abs() + log_factor.abs())
            + expected.rounding();
        assert!(
            expected.value.abs() > bar,
            "the exact log Bayes factor {} is below its bar {bar}",
            expected.value
        );
        assert!(
            (log_factor - expected.value).abs() <= bar,
            "log BF {log_factor} vs {} within {bar}",
            expected.value
        );
    }

    #[test]
    fn function_prior_preserves_extreme_precision_products_and_student_scale_units() {
        let (model, histories, _) = fixture();
        let prior = bounded(&model, &histories);
        let position = |label: FunctionPenalty| {
            prior
                .penalties()
                .iter()
                .position(|existing| *existing == label)
                .unwrap()
        };
        let drive = position(FunctionPenalty::DriveLevel);
        for (strength, coefficient) in [(800.0_f64, 1e-200_f64), (-800.0, 1e200)] {
            let start = model.layout.drive.start;
            let mut values = vec![0.0; model.layout.width];
            values[start] = coefficient;
            let mut rho_values = vec![0.0; prior.penalties().len()];
            rho_values[drive] = strength;
            let out = prior.evaluate(&exact(&values), &exact(&rho_values)).unwrap();
            // The mean-level block (1, mu)(1, mu)' has entry 1 at the level coordinate and the
            // dependence block none there, so the gradient is -exp(strength) coefficient.
            let expected = exp(&Bound::exact(strength).add(&ln(&Bound::exact(coefficient)))).neg();
            agrees(&out.gradient()[start], &expected, &format!("gradient at {strength}"));
            let mut direction = vec![Bound::exact(0.0); values.len()];
            direction[start] = Bound::exact(coefficient);
            let product = out.negative_hessian_product(&direction).unwrap();
            agrees(&product[start], &expected.neg(), &format!("product at {strength}"));
        }
        let f64_prior = model
            .function_priors(&histories.iter().collect::<Vec<_>>())
            .unwrap();
        // A precision whose square root overflows against a coefficient is refused, not rounded:
        // exp(1420/2) times a unit drive level is not representable.
        let mut rho_values = vec![0.0; prior.penalties().len()];
        rho_values[drive] = 1420.0;
        let mut unit_level = vec![0.0; model.layout.width];
        unit_level[model.layout.drive.start] = 1.0;
        assert!(matches!(
            f64_prior.evaluate(&unit_level, &rho_values),
            Err(EventHistoryError::NumericalFailure { .. })
        ));
        // rho/2 - log scale = 0 exactly at (rho, log scale) = (0, 0) and (800, 400), so every
        // Student-t block has bitwise-identical energy and gradients.
        let mut theta = vec![0.0; model.layout.width];
        let location = model.layout.measurement_location[0].start;
        theta[location + 1] = 0.4;
        theta[location + 3] = 0.7;
        let mut rho = vec![0.0; prior.penalties().len()];
        let standard = f64_prior.evaluate(&theta, &rho).unwrap();
        for label in [
            FunctionPenalty::MeasurementEffect { channel: 0 },
            FunctionPenalty::MeasurementVariation { penalty: 0 },
            FunctionPenalty::MeasurementVariation { penalty: 1 },
        ] {
            rho[position(label)] = 800.0;
        }
        theta[model.layout.measurement_shape[0].start] = 400.0;
        let shifted = f64_prior.evaluate(&theta, &rho).unwrap();
        let mut compared = 0;
        for (index, function) in f64_prior.gaussian.iter().enumerate() {
            if function.log_scale.is_some() {
                assert_eq!(
                    standard.gaussian[index].weighted_penalty,
                    shifted.gaussian[index].weighted_penalty
                );
                assert_eq!(standard.gaussian[index].gradients, shifted.gaussian[index].gradients);
                assert!(standard.gaussian[index].weighted_penalty > 0.0);
                compared += 1;
            }
        }
        // The loading level and both variation blocks of the Student-t channel.
        assert_eq!(compared, 3);
    }
}

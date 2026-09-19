//! #977 / #907 — cross-class shape adjudication on a recovered atom's
//! intrinsic 2-D coordinates (the SAME Rust evidence code the in-tree
//! `quality_llm_weekday_circle` gate drives), exposed so the real-activation
//! driver computes the verdict with ONE evidence implementation, not a Python
//! re-implementation. Races a smooth S¹ ring, a Euclidean Gaussian, a free
//! k-cluster mixture, and a circle-constrained ring of clusters; the held-out
//! predictive-stacking headline picks the winner.

use gam_solve::CircularGaussianFit2d;
use ndarray::{Array2, ArrayView2};

/// Held-out log-density of the smooth-circle (ring) candidate on 2-D coords:
/// a uniform latent point on the fitted circle convolved with isotropic 2-D
/// Gaussian noise. This is a normalized Cartesian density, including at the
/// fitted center.
fn ring_provider_2d(coords: Array2<f64>) -> gam_solve::HeldOutDensityProvider<'static> {
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let (train_coords, eval_coords, log_volume_scale) =
                canonical_shape_fold(coords.view(), train, eval, "circle density")?;
            let train_rows = (0..train_coords.nrows()).collect::<Vec<_>>();
            let fit = CircularGaussianFit2d::fit(train_coords.view(), &train_rows)?;
            let mut out = Vec::with_capacity(eval_coords.nrows());
            for row in eval_coords.rows() {
                out.push(fit.log_density(row[0], row[1]) - log_volume_scale);
            }
            Ok(out)
        },
    )
}

/// One full Euclidean Gaussian fit in two dimensions.
///
/// `origin + mean_offset` is the location, represented in two pieces so a
/// translated intrinsic chart does not force us to subtract two large,
/// nearly-equal absolute coordinates when accumulating or evaluating the fit.
/// The covariance is constrained spectrally, and `precision` plus `log_norm`
/// are constructed from those exact same constrained eigenvalues.  This is
/// essential: independently flooring covariance entries and its determinant
/// does not describe any normalized Gaussian density.
#[derive(Debug, Clone, Copy)]
struct GaussianFit2d {
    origin: [f64; 2],
    mean_offset: [f64; 2],
    major_direction: [f64; 2],
    inverse_eigenvalues: [f64; 2],
    log_norm: f64,
}

impl GaussianFit2d {
    fn fit(coords: ArrayView2<'_, f64>, rows: &[usize]) -> Result<Self, String> {
        if coords.ncols() != 2 || rows.iter().any(|&row| row >= coords.nrows()) {
            return Err("Gaussian density received invalid coordinates or row indices".to_string());
        }
        if rows.len() < 3 {
            return Err("Gaussian density needs at least three training rows".to_string());
        }
        let origin = [coords[[rows[0], 0]], coords[[rows[0], 1]]];
        if !origin.iter().all(|value| value.is_finite()) {
            return Err("Gaussian density requires finite coordinates".to_string());
        }

        let mut mean_offset = [0.0_f64; 2];
        for &row in rows {
            for axis in 0..2 {
                let relative = coords[[row, axis]] - origin[axis];
                if !relative.is_finite() {
                    return Err("Gaussian density requires finite coordinates".to_string());
                }
                mean_offset[axis] += relative;
            }
        }
        let count = rows.len() as f64;
        mean_offset[0] /= count;
        mean_offset[1] /= count;

        let (mut sxx, mut sxy, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
        for &row in rows {
            let dx = (coords[[row, 0]] - origin[0]) - mean_offset[0];
            let dy = (coords[[row, 1]] - origin[1]) - mean_offset[1];
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        sxx /= count;
        sxy /= count;
        syy /= count;
        let trace = sxx + syy;
        if !(trace.is_finite() && trace > 0.0) || !sxy.is_finite() {
            return Err(
                "Gaussian density is undefined for a point cloud with zero or non-finite variance"
                    .to_string(),
            );
        }

        // For [[sxx,sxy],[sxy,syy]], theta is the major-eigenvector angle.
        // The relative spectral floor is homogeneous in the data units: under
        // x -> a*x both eigenvalues and the floor multiply by a^2.  The old
        // absolute entry/determinant floors changed the model under this
        // harmless re-expression of an intrinsic coordinate chart.
        let spectral_gap = (sxx - syy).hypot(2.0 * sxy);
        let largest = (0.5 * (trace + spectral_gap)).max(f64::MIN_POSITIVE);
        // `sxx`, `syy` are means of `count` nonnegative squares of centered offsets,
        // each formed with at most three rounded operations, so each is off by
        // `γ_{count+4}` of itself; `sxy` is off by `γ_{count+4}·(sxx + syy)/2`
        // (AM–GM). `trace` and the gap `hypot(sxx − syy, 2·sxy) ≤ trace` inherit
        // those errors and a few more roundings, so the cancelling smallest
        // eigenvalue `½·(trace − gap)` is resolved only to `γ_{3·count+17}·trace`.
        // An exact smallest eigenvalue inside that band cannot be told from zero.
        let floor = (gam_linalg::roundoff::accumulation_growth(3 * rows.len() + 17) * trace)
            .max(f64::MIN_POSITIVE);
        let smallest = (0.5 * (trace - spectral_gap)).max(floor);
        let largest = largest.max(floor);
        if !smallest.is_finite() || !largest.is_finite() {
            return Err("Gaussian covariance spectrum is non-finite".to_string());
        }
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let (sin_theta, cos_theta) = theta.sin_cos();
        let inverse_largest = 1.0 / largest;
        let inverse_smallest = 1.0 / smallest;
        let log_norm = -std::f64::consts::TAU.ln() - 0.5 * (largest.ln() + smallest.ln());
        if !inverse_largest.is_finite() || !inverse_smallest.is_finite() || !log_norm.is_finite() {
            return Err("Gaussian covariance factorization is non-finite".to_string());
        }
        Ok(Self {
            origin,
            mean_offset,
            major_direction: [cos_theta, sin_theta],
            inverse_eigenvalues: [inverse_largest, inverse_smallest],
            log_norm,
        })
    }

    fn log_density(self, x: f64, y: f64) -> f64 {
        let dx = (x - self.origin[0]) - self.mean_offset[0];
        let dy = (y - self.origin[1]) - self.mean_offset[1];
        // Evaluate in the covariance eigenbasis.  This sum of nonnegative
        // squares avoids the cancellation of an expanded x' Sigma^-1 x for a
        // nearly rank-one cloud.
        let major = self.major_direction[0] * dx + self.major_direction[1] * dy;
        let minor = -self.major_direction[1] * dx + self.major_direction[0] * dy;
        let quad = major * major * self.inverse_eigenvalues[0]
            + minor * minor * self.inverse_eigenvalues[1];
        self.log_norm - 0.5 * quad
    }
}

/// Held-out log-density of the Euclidean candidate: a full 2-D Gaussian (mean +
/// 2×2 covariance) refit on each fold's training rows.
fn gaussian_provider_2d(coords: Array2<f64>) -> gam_solve::HeldOutDensityProvider<'static> {
    Box::new(
        move |train: &[usize], eval: &[usize]| -> Result<Vec<f64>, String> {
            let (train_coords, eval_coords, log_volume_scale) =
                canonical_shape_fold(coords.view(), train, eval, "Gaussian density")?;
            let train_rows = (0..train_coords.nrows()).collect::<Vec<_>>();
            let fit = GaussianFit2d::fit(train_coords.view(), &train_rows)?;
            let mut out = Vec::with_capacity(eval_coords.nrows());
            for row in eval_coords.rows() {
                out.push(fit.log_density(row[0], row[1]) - log_volume_scale);
            }
            Ok(out)
        },
    )
}

/// BIC/2 of the circular Gaussian (4 parameters: center(2), circle radius, and
/// isotropic noise variance). Corroborates the held-out stacking headline;
/// lower is better.
fn ring_bic_2d(coords: ArrayView2<'_, f64>) -> Result<f64, String> {
    let n = coords.nrows();
    let rows = (0..n).collect::<Vec<_>>();
    let (_, bic) = CircularGaussianFit2d::fit_with_bic(coords, &rows)?;
    Ok(bic)
}

/// BIC/2 of the full 2-D Gaussian (5 free params:
/// mean(2) + symmetric 2×2 covariance(3)), evaluated under the exact same
/// fitted density as the held-out provider.
fn gaussian_bic_2d(coords: ArrayView2<'_, f64>) -> Result<f64, String> {
    let n = coords.nrows();
    let rows = (0..n).collect::<Vec<_>>();
    let fit = GaussianFit2d::fit(coords, &rows)?;
    let mut loglik = 0.0_f64;
    for row in 0..n {
        loglik += fit.log_density(coords[[row, 0]], coords[[row, 1]]);
    }
    Ok(-loglik + 0.5 * 5.0 * (n as f64).ln())
}

#[derive(Debug, Clone)]
pub struct AtomShapeRaceVerdict {
    pub winner_class: String,
    pub reporting_winner: String,
    pub candidate_names: Vec<String>,
    pub stacking_weights: Vec<f64>,
    pub bic: Vec<f64>,
    pub mixture_reporting_k: usize,
    pub ring_clusters_reporting_k: usize,
    pub mixture_fold_selected_k: Vec<usize>,
    pub ring_clusters_fold_selected_k: Vec<usize>,
    pub mixture_fold_k_histogram: std::collections::BTreeMap<usize, usize>,
    pub ring_clusters_fold_k_histogram: std::collections::BTreeMap<usize, usize>,
    pub circular_stacking_weight: f64,
    pub noncircular_stacking_weight: f64,
    pub circular_margin: f64,
    pub circle_wins: bool,
    pub is_cross_class: bool,
    pub headline: &'static str,
}

fn circular_stacking_summary(
    candidate_kinds: &[gam_solve::PredictiveCandidateKind],
    stacking_weights: &[f64],
    simplex_residual: f64,
) -> Result<(f64, f64, f64, bool), String> {
    if candidate_kinds.len() != stacking_weights.len() || candidate_kinds.is_empty() {
        return Err(format!(
            "shape stacking result has {} candidate kinds but {} weights",
            candidate_kinds.len(),
            stacking_weights.len()
        ));
    }
    let mut circular_weight = 0.0_f64;
    let mut noncircular_weight = 0.0_f64;
    for (&kind, &weight) in candidate_kinds.iter().zip(stacking_weights) {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "shape stacking returned invalid weight {weight} for candidate {}",
                kind.display_name()
            ));
        }
        if kind.is_circular() {
            circular_weight += weight;
        } else {
            noncircular_weight += weight;
        }
    }
    let total = circular_weight + noncircular_weight;
    // The solver certified its own sum of the `K` weights within `simplex_residual`
    // of 1, and that sum lies within `γ_K` of the exact mass. Regrouping the same
    // weights into two partial sums and adding them rounds `K + 1` more times, so an
    // honest simplex vector lands within `simplex_residual + γ_{2K+2}·(total + 1)`.
    let mass_band = simplex_residual
        + gam_linalg::roundoff::accumulation_growth(2 * stacking_weights.len() + 2)
            * (total + 1.0);
    if !total.is_finite() || !simplex_residual.is_finite() || (total - 1.0).abs() > mass_band {
        return Err(format!(
            "shape stacking weights must have unit mass; got {total} against the certified \
             simplex residual {simplex_residual}"
        ));
    }
    let circular_margin = circular_weight - noncircular_weight;
    Ok((
        circular_weight,
        noncircular_weight,
        circular_margin,
        circular_margin > 0.0,
    ))
}

#[derive(Clone, Copy)]
struct ShapeCoordinateGauge {
    coordinate_scale: f64,
    mean_scaled: [f64; 2],
    rms_scaled: f64,
    log_scale: f64,
}

/// Fit the translation and uniform-scale gauge from training rows only.
///
/// Coordinates are first divided by their largest absolute training value.
/// Thus every subsequent subtraction lies in a bounded chart even for finite
/// antipodal values whose raw difference exceeds float64. The mean is a
/// sequence of convex combinations and the RMS uses the LAPACK `lassq`
/// recurrence rather than summing raw squares. The physical scale is retained
/// in log space, so its Jacobian remains representable even if the product of
/// the two chart scales would overflow or underflow.
fn fit_shape_coordinate_gauge(
    training: ArrayView2<'_, f64>,
    context: &str,
) -> Result<ShapeCoordinateGauge, String> {
    if training.ncols() != 2
        || training.nrows() == 0
        || !training.iter().all(|value| value.is_finite())
    {
        return Err(format!(
            "{context} training coordinates must be a nonempty finite (n, 2) matrix"
        ));
    }
    let coordinate_scale = training
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !(coordinate_scale.is_finite() && coordinate_scale > 0.0) {
        return Err(format!(
            "{context} training coordinates have zero or non-finite scale"
        ));
    }
    let mut mean_scaled = [0.0_f64; 2];
    for row in 0..training.nrows() {
        let count = (row + 1) as f64;
        let previous_weight = (count - 1.0) / count;
        let new_weight = 1.0 / count;
        for axis in 0..2 {
            let scaled = training[[row, axis]] / coordinate_scale;
            mean_scaled[axis] = previous_weight * mean_scaled[axis] + new_weight * scaled;
        }
    }

    let mut norm_scale = 0.0_f64;
    let mut scaled_sum_squares = 1.0_f64;
    for row in training.rows() {
        for axis in 0..2 {
            let centered = row[axis] / coordinate_scale - mean_scaled[axis];
            if !centered.is_finite() {
                return Err(format!(
                    "{context} centered training coordinate overflowed on axis {axis}"
                ));
            }
            let magnitude = centered.abs();
            if magnitude == 0.0 {
                continue;
            }
            if norm_scale < magnitude {
                scaled_sum_squares =
                    1.0 + scaled_sum_squares * (norm_scale / magnitude) * (norm_scale / magnitude);
                norm_scale = magnitude;
            } else {
                scaled_sum_squares += (magnitude / norm_scale) * (magnitude / norm_scale);
            }
        }
    }
    let rms_scaled = norm_scale * (scaled_sum_squares / training.nrows() as f64).sqrt();
    let log_scale = coordinate_scale.ln() + rms_scaled.ln();
    if !(rms_scaled.is_finite() && rms_scaled > 0.0 && log_scale.is_finite()) {
        return Err(format!(
            "{context} training coordinates have zero or non-finite centered scale"
        ));
    }
    Ok(ShapeCoordinateGauge {
        coordinate_scale,
        mean_scaled,
        rms_scaled,
        log_scale,
    })
}

fn apply_shape_coordinate_gauge(
    coords: ArrayView2<'_, f64>,
    gauge: ShapeCoordinateGauge,
    context: &str,
) -> Result<Array2<f64>, String> {
    if coords.ncols() != 2 || !coords.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "{context} coordinates must be a finite (n, 2) matrix"
        ));
    }
    let mut canonical = Array2::<f64>::zeros(coords.raw_dim());
    for row in 0..coords.nrows() {
        for axis in 0..2 {
            let value = (coords[[row, axis]] / gauge.coordinate_scale - gauge.mean_scaled[axis])
                / gauge.rms_scaled;
            if !value.is_finite() {
                return Err(format!(
                    "{context} canonical coordinate overflowed at row {row}, axis {axis}"
                ));
            }
            canonical[[row, axis]] = value;
        }
    }
    Ok(canonical)
}

/// Canonicalize one reporting fit from all of its rows. Outer-CV providers use
/// [`canonical_shape_fold`] instead so held-out rows cannot choose their gauge.
fn canonical_shape_coordinates(coords: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let gauge = fit_shape_coordinate_gauge(coords, "shape")?;
    apply_shape_coordinate_gauge(coords, gauge, "shape")
}

fn gather_shape_rows(
    coords: ArrayView2<'_, f64>,
    rows: &[usize],
    context: &str,
) -> Result<Array2<f64>, String> {
    if rows.iter().any(|&row| row >= coords.nrows()) {
        return Err(format!(
            "{context} contains an out-of-bounds row for {} coordinates",
            coords.nrows()
        ));
    }
    let mut gathered = Array2::<f64>::zeros((rows.len(), coords.ncols()));
    for (target, &source) in rows.iter().enumerate() {
        gathered.row_mut(target).assign(&coords.row(source));
    }
    Ok(gathered)
}

/// Derive one candidate-common chart from an outer fold's training rows and
/// apply it to both train and evaluation rows. Densities are fitted in this
/// dimensionless chart; `log_volume_scale = log(scale²)` converts their scores
/// back to proper densities in the caller's original coordinate units.
fn canonical_shape_fold(
    coords: ArrayView2<'_, f64>,
    train: &[usize],
    eval: &[usize],
    context: &str,
) -> Result<(Array2<f64>, Array2<f64>, f64), String> {
    let training = gather_shape_rows(coords, train, &format!("{context} training fold"))?;
    let evaluation = gather_shape_rows(coords, eval, &format!("{context} evaluation fold"))?;
    let gauge = fit_shape_coordinate_gauge(training.view(), context)?;
    let log_volume_scale = 2.0 * gauge.log_scale;
    if !log_volume_scale.is_finite() {
        return Err(format!("{context} coordinate Jacobian is non-finite"));
    }
    Ok((
        apply_shape_coordinate_gauge(training.view(), gauge, context)?,
        apply_shape_coordinate_gauge(evaluation.view(), gauge, context)?,
        log_volume_scale,
    ))
}

/// Select the free-cluster order using only an outer fold's training rows, then
/// score that fold's untouched evaluation rows. The full-data order remains a
/// useful final-fit/evidence summary, but it must never choose the model used to
/// construct an outer-held-out predictive density.
fn free_mixture_rung_predictive_density(
    train: ArrayView2<'_, f64>,
    eval: ArrayView2<'_, f64>,
    config: gam_solve::evidence::GaussianMixtureConfig,
) -> Result<(usize, Vec<f64>), String> {
    let rung = gam_solve::fit_free_cluster_rung(train, config)
        .map_err(|error| error.to_string())?;
    let fit = rung.winner();
    Ok((fit.k, fit.fit.per_point_log_density(eval)?.to_vec()))
}

fn ring_cluster_rung_predictive_density(
    train: ArrayView2<'_, f64>,
    eval: ArrayView2<'_, f64>,
    config: gam_solve::evidence::GaussianMixtureConfig,
) -> Result<(usize, Vec<f64>), String> {
    let rung = gam_solve::fit_ring_of_clusters_rung(train, config)
        .map_err(|error| error.to_string())?;
    let fit = rung.winner();
    Ok((fit.k, fit.fit.per_point_log_density(eval)?.to_vec()))
}

fn free_mixture_rung_provider_2d(
    coords: Array2<f64>,
    config: gam_solve::evidence::GaussianMixtureConfig,
    selected_orders: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
) -> gam_solve::HeldOutDensityProvider<'static> {
    Box::new(move |train: &[usize], eval: &[usize]| {
        let (train_coords, eval_coords, log_volume_scale) =
            canonical_shape_fold(coords.view(), train, eval, "mixture density")?;
        let (selected_k, mut density) = free_mixture_rung_predictive_density(
            train_coords.view(),
            eval_coords.view(),
            config,
        )?;
        for value in &mut density {
            *value -= log_volume_scale;
        }
        selected_orders.borrow_mut().push(selected_k);
        Ok(density)
    })
}

fn ring_cluster_rung_provider_2d(
    coords: Array2<f64>,
    config: gam_solve::evidence::GaussianMixtureConfig,
    selected_orders: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
) -> gam_solve::HeldOutDensityProvider<'static> {
    Box::new(move |train: &[usize], eval: &[usize]| {
        let (train_coords, eval_coords, log_volume_scale) =
            canonical_shape_fold(coords.view(), train, eval, "ring-cluster density")?;
        let (selected_k, mut density) = ring_cluster_rung_predictive_density(
            train_coords.view(),
            eval_coords.view(),
            config,
        )?;
        for value in &mut density {
            *value -= log_volume_scale;
        }
        selected_orders.borrow_mut().push(selected_k);
        Ok(density)
    })
}

fn finish_fold_order_trace(
    selected_orders: &std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
    folds: usize,
    class_name: &str,
) -> Result<Vec<usize>, String> {
    let orders = selected_orders.borrow().clone();
    if orders.len() != folds {
        return Err(format!(
            "{class_name} recorded {} fold-local orders for {folds} folds",
            orders.len()
        ));
    }
    Ok(orders)
}

fn fold_order_histogram(orders: &[usize]) -> std::collections::BTreeMap<usize, usize> {
    let mut histogram = std::collections::BTreeMap::new();
    for &order in orders {
        *histogram.entry(order).or_insert(0) += 1;
    }
    histogram
}

pub fn run_atom_shape_race(
    coords: ArrayView2<'_, f64>,
    folds: usize,
    seed: u64,
) -> Result<AtomShapeRaceVerdict, String> {
    use gam_solve::evidence::{GaussianMixtureConfig, StackingConfig};
    use gam_solve::topology_selector::EvidenceCertification;
    use gam_solve::{
        AutoTopologyKind, Headline, PredictiveCandidateKind, PredictiveRaceCandidate,
        adjudicate_predictive_race, fit_free_cluster_rung, fit_ring_of_clusters_rung,
    };

    if coords.ncols() != 2 {
        return Err(format!(
            "adjudicate_atom_shape: coords must be (n, 2); got {:?}",
            coords.dim()
        ));
    }
    let n = coords.nrows();
    if n < 4 {
        return Err("adjudicate_atom_shape: need at least 4 rows to adjudicate".to_string());
    }
    if folds < 2 || folds > n {
        return Err(format!(
            "adjudicate_atom_shape: require 2 <= folds <= n; got folds={folds}, n={n}"
        ));
    }
    let largest_evaluation_fold = n / folds + usize::from(n % folds != 0);
    let minimum_training_rows = n - largest_evaluation_fold;
    if minimum_training_rows < 3 {
        return Err(format!(
            "adjudicate_atom_shape: every outer training fold must contain at least 3 rows for the ring-cluster class; got a minimum of {minimum_training_rows} with n={n}, folds={folds}"
        ));
    }
    if !coords.iter().all(|value| value.is_finite()) {
        return Err("adjudicate_atom_shape: coords must be finite".to_string());
    }
    // Full-data coordinates are canonicalized only for reporting fits and
    // corroborating BIC/2 scores. Every outer-CV provider below receives the raw
    // chart and derives its gauge from that fold's training rows alone.
    let reporting_coords = canonical_shape_coordinates(coords)?;
    let raw_coords = coords.to_owned();
    let config = GaussianMixtureConfig::default();
    // `Euclidean` already is the one-component full Gaussian. Letting the
    // mixture rung choose k=1 inserts the identical predictive density twice,
    // so the stacking optimum is non-identifiable and the reported weights
    // depend on candidate ordering. The free *cluster* contender begins at two
    // components; the circular cluster model begins at three.
    let mixture = fit_free_cluster_rung(reporting_coords.view(), config)
        .map_err(|error| error.to_string())?;
    let mixture_winner = mixture.winner();
    let mixture_reporting_k = mixture_winner.k;
    let ring_clusters = fit_ring_of_clusters_rung(reporting_coords.view(), config)
        .map_err(|error| error.to_string())?;
    let ring_clusters_reporting_k = ring_clusters.winner().k;
    let mixture_fold_orders = std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(folds)));
    let ring_cluster_fold_orders =
        std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(folds)));
    let candidate_kinds = [
        PredictiveCandidateKind::Fixed(AutoTopologyKind::Circle),
        PredictiveCandidateKind::Fixed(AutoTopologyKind::Euclidean),
        PredictiveCandidateKind::MixtureClass,
        PredictiveCandidateKind::RingOfClustersClass,
    ];
    let candidates = vec![
        PredictiveRaceCandidate {
            kind: candidate_kinds[0],
            bic_half: ring_bic_2d(reporting_coords.view())?,
            certification: EvidenceCertification::Exact,
            density_provider: ring_provider_2d(raw_coords.clone()),
        },
        PredictiveRaceCandidate {
            kind: candidate_kinds[1],
            bic_half: gaussian_bic_2d(reporting_coords.view())?,
            certification: EvidenceCertification::Exact,
            density_provider: gaussian_provider_2d(raw_coords.clone()),
        },
        PredictiveRaceCandidate {
            kind: candidate_kinds[2],
            bic_half: mixture_winner.bic,
            certification: EvidenceCertification::Exact,
            // The displayed/reported k is the full-data final fit. Its outer-CV
            // predictive column independently selects k on each training fold,
            // and derives its chart gauge from those same rows, so held-out
            // rows cannot leak into either preprocessing or model selection.
            density_provider: free_mixture_rung_provider_2d(
                raw_coords.clone(),
                config,
                std::rc::Rc::clone(&mixture_fold_orders),
            ),
        },
        PredictiveRaceCandidate {
            kind: candidate_kinds[3],
            bic_half: ring_clusters.winner().bic,
            certification: EvidenceCertification::Exact,
            density_provider: ring_cluster_rung_provider_2d(
                raw_coords,
                config,
                std::rc::Rc::clone(&ring_cluster_fold_orders),
            ),
        },
    ];
    let verdict =
        adjudicate_predictive_race(n, candidates, folds, seed, StackingConfig::default())?;
    let (stacking_weights, simplex_residual) = verdict
        .stacking
        .as_ref()
        .map(|stacking| (stacking.weights.to_vec(), stacking.certificate.simplex_residual))
        .ok_or_else(|| {
            "shape race mixed model classes but returned no stacking result".to_string()
        })?;
    let winner_class = candidate_kinds[verdict.winner_index]
        .family_tag()
        .to_string();
    let reporting_winner = match candidate_kinds[verdict.winner_index] {
        PredictiveCandidateKind::MixtureClass => format!("mixture_k{mixture_reporting_k}"),
        PredictiveCandidateKind::RingOfClustersClass => {
            format!("ring_clusters_k{ring_clusters_reporting_k}")
        }
        kind => kind.display_name(),
    };
    let (circular_stacking_weight, noncircular_stacking_weight, circular_margin, circle_wins) =
        circular_stacking_summary(&candidate_kinds, &stacking_weights, simplex_residual)?;
    let mixture_fold_selected_k =
        finish_fold_order_trace(&mixture_fold_orders, folds, "mixture class")?;
    let ring_clusters_fold_selected_k =
        finish_fold_order_trace(&ring_cluster_fold_orders, folds, "ring-cluster class")?;
    let mixture_fold_k_histogram = fold_order_histogram(&mixture_fold_selected_k);
    let ring_clusters_fold_k_histogram = fold_order_histogram(&ring_clusters_fold_selected_k);
    Ok(AtomShapeRaceVerdict {
        circle_wins,
        winner_class,
        reporting_winner,
        candidate_names: verdict.candidate_names,
        stacking_weights,
        bic: verdict.bic_half,
        mixture_reporting_k,
        ring_clusters_reporting_k,
        mixture_fold_selected_k,
        ring_clusters_fold_selected_k,
        mixture_fold_k_histogram,
        ring_clusters_fold_k_histogram,
        circular_stacking_weight,
        noncircular_stacking_weight,
        circular_margin,
        is_cross_class: verdict.is_cross_class,
        headline: match verdict.headline {
            Headline::Stacking => "stacking",
            Headline::Evidence => "evidence",
        },
    })
}

/// Run the two matched structureless controls (#2262) for one shape
/// adjudication and return `(shuffle_verdict, gaussian_verdict,
/// control_circular_win_fraction)`. The last value is descriptive across the
/// two controls, not an estimated false-positive rate. Pulled out of the
/// pyfunction body so it is directly unit-testable without Python.
pub fn matched_control_verdicts(
    coords_view: ArrayView2<'_, f64>,
    folds: usize,
    seed: u64,
    mean_l0: Option<f64>,
) -> Result<(AtomShapeRaceVerdict, AtomShapeRaceVerdict, f64), String> {
    validate_control_mean_l0(mean_l0)?;
    use crate::null_battery::{
        covariance_matched_gaussian_null, per_dimension_shuffle_null,
    };
    let shuffled = per_dimension_shuffle_null(coords_view, seed ^ 0xD1AE_510F)?;
    let gaussian = covariance_matched_gaussian_null(coords_view, seed ^ 0xC0A4_71A1)?;
    let shuffle_verdict = run_atom_shape_race(shuffled.view(), folds, seed)?;
    let gaussian_verdict = run_atom_shape_race(gaussian.view(), folds, seed)?;
    let control_circular_win_fraction = (usize::from(shuffle_verdict.circle_wins)
        + usize::from(gaussian_verdict.circle_wins)) as f64
        / 2.0;
    Ok((
        shuffle_verdict,
        gaussian_verdict,
        control_circular_win_fraction,
    ))
}

pub fn validate_control_mean_l0(mean_l0: Option<f64>) -> Result<f64, String> {
    let mean_l0 = mean_l0.ok_or_else(|| {
        "adjudicate_atom_shape: mean_l0 is required when matched_controls=True; a shape-verdict rate without dictionary sparsity is uninterpretable"
            .to_string()
    })?;
    if !mean_l0.is_finite() || mean_l0 < 0.0 {
        return Err(format!(
            "adjudicate_atom_shape: mean_l0 must be finite and non-negative; got {mean_l0}"
        ));
    }
    Ok(mean_l0)
}

pub fn shape_reconstruction_rank_edge(
    n_eff: Option<f64>,
    ambient_p: Option<f64>,
    dispersion_r: Option<f64>,
) -> Result<Option<f64>, String> {
    match (n_eff, ambient_p, dispersion_r) {
        (None, None, None) => Ok(None),
        (Some(n_eff), Some(ambient_p), Some(dispersion_r)) => {
            crate::null_battery::mp_reconstruction_rank_edge(
                n_eff,
                ambient_p,
                dispersion_r,
            )
            .map(Some)
        }
        _ => Err(
            "adjudicate_atom_shape: n_eff, ambient_p, and dispersion_r must be supplied together"
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod atom_shape_race_tests;

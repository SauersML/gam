//! Weight slices on an atlas of a use site's input manifold (#2951).
//!
//! # The object
//!
//! A weight `W ∈ ℝ^{q×p}` at a use site reads one input stream `z ∈ ℝ^p`. Its data
//! live on a manifold `M ⊂ ℝ^p`, and on a neighbourhood of `M` only a slice of `W` is
//! used. An atlas of `M` is a finite partition of the observed rows into cells, each
//! carrying a chart: a mean `μ_c` and a `d`-dimensional affine patch through it. On a
//! row in cell `c` the weights used are the chart's slice
//!
//! ```text
//! z ↦ W μ_c + O_c O_cᵀ W (z − μ_c),        O_c ∈ ℝ^{q×d},  O_cᵀ O_c = I_d,
//! ```
//!
//! a rank-`d` affine map, so the decomposition of `W` is
//! `W z = Σ_c 1[z ∈ c] [W μ_c + O_c O_cᵀ W (z − μ_c)] + (normal residual)`: a partition
//! of the parameters by the geometry of the data they act on. No component is rank one,
//! and nothing is penalized: each chart is the closed-form optimum below.
//!
//! # The slice is reduced-rank regression, exactly
//!
//! Over a cell's rows `Z_c` (`n_c × p`), the rank-`d` affine slice minimizing the
//! summed output error `Σ_{z ∈ c} ‖W z − ŷ(z)‖²` among maps
//! `ŷ(z) = W μ_c + L (z − μ_c)` with `rank L ≤ d` is, by Eckart–Young applied to
//! `Y_c = (Z_c − 1 μ_cᵀ) Wᵀ`, `L = O_c O_cᵀ W` with `O_c` the top `d` right singular
//! vectors of `Y_c`, and the minimum equals `Σ_{i > d} σ_i(Y_c)²` exactly
//! ([`ChartSlice::bank_residual`]). The mean is optimal because the slice is affine:
//! the centred error is orthogonal to constants. The criterion is on the site's
//! OUTPUT `W z`, never on input variance: an input direction with large spread in the
//! null space of `W` costs nothing and is never kept, which input principal components
//! cannot see (`nullspace_variance_is_ignored` in the tests).
//!
//! # Execution through the input
//!
//! A framework executes the slice by replacing the input: `ẑ = μ_c + Aᵀ R (z − μ_c)`
//! with read `R = O_cᵀ W` (`d × p`) and write `Aᵀ = W⁺ O_c` (`p × d`). The columns of
//! `O_c` lie in `range(W)` (they are right singular vectors of `Y_c`, whose rows are in
//! `range(W)`), and `W W⁺` is the projector onto `range(W)`, so `W ẑ` equals the slice
//! exactly. The pseudo-inverse is taken on the singular values of `W` resolved from
//! zero by the backward band `max(q, p)·ε·σ₁`; a direction of `O_c` outside the resolved
//! range would make the write inexact and is refused.
//!
//! # Cells
//!
//! [`lloyd_cells`] partitions the rows by Lloyd's iteration from farthest-point seeds:
//! the first seed is the row nearest the bank mean, and each next seed is the row
//! farthest from the seeds so far, so the partition has no random draw. Each sweep
//! assigns every row to its nearest centre and moves every centre to its cell's mean;
//! the within-cell sum of squares never increases and there are finitely many
//! partitions, so the iteration reaches a fixed point, where it stops. A cell that
//! empties keeps its centre. The result is a converged partition, not a budgeted one.
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerSvd};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::fmt;

/// Why a slice atlas could not be built.
#[derive(Debug)]
pub enum ChartSliceError {
    /// The bank has no rows.
    EmptyBank,
    /// The weight's input width does not match the bank's.
    WidthMismatch { bank: usize, weight: usize },
    /// A bank or weight entry is not finite.
    NonFinite { what: &'static str, row: usize, col: usize },
    /// More charts were requested than there are rows.
    TooManyCharts { charts: usize, rows: usize },
    /// No chart was requested.
    NoCharts,
    /// The assignment has the wrong length or names a chart out of range.
    InvalidAssignment { index: usize },
    /// The slice rank exceeds the weight's output width.
    RankTooLarge { rank: usize, output_width: usize },
    /// A kept output direction is not in the resolved range of the weight, so the
    /// input-side write cannot reproduce it exactly.
    OutsideResolvedRange { chart: usize, leakage: f64, band: f64 },
    /// The SVD owner failed.
    Svd(FaerLinalgError),
}

impl fmt::Display for ChartSliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBank => write!(f, "chart_slice: the bank has no rows"),
            Self::WidthMismatch { bank, weight } => {
                write!(f, "chart_slice: bank width {bank} but the weight reads {weight}")
            }
            Self::NonFinite { what, row, col } => {
                write!(f, "chart_slice: {what}[{row}, {col}] is not finite")
            }
            Self::TooManyCharts { charts, rows } => {
                write!(f, "chart_slice: {charts} charts requested for {rows} rows")
            }
            Self::NoCharts => write!(f, "chart_slice: no chart requested"),
            Self::InvalidAssignment { index } => {
                write!(f, "chart_slice: assignment entry {index} is missing or out of range")
            }
            Self::RankTooLarge { rank, output_width } => write!(
                f,
                "chart_slice: slice rank {rank} exceeds the weight's output width {output_width}"
            ),
            Self::OutsideResolvedRange { chart, leakage, band } => write!(
                f,
                "chart_slice: chart {chart} keeps an output direction outside the weight's \
                 resolved range (leakage {leakage:.3e} > band {band:.3e})"
            ),
            Self::Svd(error) => write!(f, "chart_slice: SVD failed: {error}"),
        }
    }
}

impl std::error::Error for ChartSliceError {}

impl From<FaerLinalgError> for ChartSliceError {
    fn from(error: FaerLinalgError) -> Self {
        Self::Svd(error)
    }
}

/// One chart: its cell mean and its rank-`d` slice of the weight.
#[derive(Clone, Debug)]
pub struct ChartSlice {
    mean: Array1<f64>,
    /// `R = O_cᵀ W`, `d × p`.
    read: Array2<f64>,
    /// `(W⁺ O_c)ᵀ`, `d × p`.
    write: Array2<f64>,
    /// All singular values of the cell's output block `Y_c`, descending.
    singular: Array1<f64>,
    members: usize,
}

impl ChartSlice {
    /// The cell mean `μ_c`.
    #[must_use]
    pub fn mean(&self) -> ArrayView1<'_, f64> {
        self.mean.view()
    }

    /// The read `O_cᵀ W` (`d × p`).
    #[must_use]
    pub fn read(&self) -> ArrayView2<'_, f64> {
        self.read.view()
    }

    /// The write `(W⁺ O_c)ᵀ` (`d × p`).
    #[must_use]
    pub fn write(&self) -> ArrayView2<'_, f64> {
        self.write.view()
    }

    /// Rows of the bank in this cell.
    #[must_use]
    pub fn members(&self) -> usize {
        self.members
    }

    /// The exact summed output error of the slice over its own cell's bank rows,
    /// `Σ_{i > d} σ_i(Y_c)²` (Eckart–Young), the minimum over every rank-`d` affine map.
    #[must_use]
    pub fn bank_residual(&self) -> f64 {
        let d = self.read.nrows();
        self.singular.iter().skip(d).map(|s| s * s).sum()
    }

    /// The cell's summed output energy about its mean, `Σ σ_i(Y_c)²`.
    #[must_use]
    pub fn bank_energy(&self) -> f64 {
        self.singular.iter().map(|s| s * s).sum()
    }

    /// `ẑ = μ_c + Aᵀ R (z − μ_c)`: the input whose image under `W` is the slice.
    #[must_use]
    pub fn restrict(&self, z: ArrayView1<'_, f64>) -> Array1<f64> {
        let centred = &z - &self.mean;
        let coords = self.read.dot(&centred);
        &self.mean + &self.write.t().dot(&coords)
    }
}

/// A partition of a use site's input rows into cells, each with its weight slice.
#[derive(Clone, Debug)]
pub struct SliceAtlas {
    centres: Array2<f64>,
    charts: Vec<ChartSlice>,
}

impl SliceAtlas {
    /// The cell centres (`C × p`) that assign a new row.
    #[must_use]
    pub fn centres(&self) -> ArrayView2<'_, f64> {
        self.centres.view()
    }

    /// The charts, in cell order.
    #[must_use]
    pub fn charts(&self) -> &[ChartSlice] {
        &self.charts
    }

    /// The cell whose centre is nearest `z` (ties to the lower index).
    #[must_use]
    pub fn chart_of(&self, z: ArrayView1<'_, f64>) -> usize {
        nearest(self.centres.view(), z)
    }

    /// The input restricted to its own chart's slice.
    #[must_use]
    pub fn restrict(&self, z: ArrayView1<'_, f64>) -> Array1<f64> {
        self.charts[self.chart_of(z)].restrict(z)
    }

    /// Summed exact bank residual over every chart.
    #[must_use]
    pub fn bank_residual(&self) -> f64 {
        self.charts.iter().map(ChartSlice::bank_residual).sum()
    }
}

fn nearest(centres: ArrayView2<'_, f64>, z: ArrayView1<'_, f64>) -> usize {
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (c, centre) in centres.outer_iter().enumerate() {
        let d: f64 = centre.iter().zip(z.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    best
}

fn check_finite(what: &'static str, a: ArrayView2<'_, f64>) -> Result<(), ChartSliceError> {
    for ((row, col), v) in a.indexed_iter() {
        if !v.is_finite() {
            return Err(ChartSliceError::NonFinite { what, row, col });
        }
    }
    Ok(())
}

/// Lloyd's iteration from farthest-point seeds, run to its fixed point. Returns the
/// centres (`C × p`) and every row's cell.
pub fn lloyd_cells(
    bank: ArrayView2<'_, f64>,
    charts: usize,
) -> Result<(Array2<f64>, Vec<usize>), ChartSliceError> {
    let (n, p) = bank.dim();
    if n == 0 {
        return Err(ChartSliceError::EmptyBank);
    }
    if charts == 0 {
        return Err(ChartSliceError::NoCharts);
    }
    if charts > n {
        return Err(ChartSliceError::TooManyCharts { charts, rows: n });
    }
    check_finite("bank", bank)?;
    let mean = bank.mean_axis(Axis(0)).expect("n > 0");
    let first = nearest(bank, mean.view());
    let mut centres = Array2::<f64>::zeros((charts, p));
    centres.row_mut(0).assign(&bank.row(first));
    let mut gap: Vec<f64> = bank
        .outer_iter()
        .map(|r| r.iter().zip(bank.row(first).iter()).map(|(a, b)| (a - b) * (a - b)).sum())
        .collect();
    for c in 1..charts {
        let far = gap
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |acc, (i, &g)| if g > acc.1 { (i, g) } else { acc })
            .0;
        centres.row_mut(c).assign(&bank.row(far));
        for (i, r) in bank.outer_iter().enumerate() {
            let d: f64 = r.iter().zip(bank.row(far).iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < gap[i] {
                gap[i] = d;
            }
        }
    }
    let mut assign: Vec<usize> = bank.outer_iter().map(|r| nearest(centres.view(), r)).collect();
    loop {
        let mut sums = Array2::<f64>::zeros((charts, p));
        let mut counts = vec![0usize; charts];
        for (i, r) in bank.outer_iter().enumerate() {
            let mut row = sums.row_mut(assign[i]);
            row += &r;
            counts[assign[i]] += 1;
        }
        for c in 0..charts {
            if counts[c] > 0 {
                let mut row = sums.row_mut(c);
                row /= counts[c] as f64;
                centres.row_mut(c).assign(&row);
            }
        }
        let next: Vec<usize> = bank.outer_iter().map(|r| nearest(centres.view(), r)).collect();
        if next == assign {
            return Ok((centres, assign));
        }
        assign = next;
    }
}

/// Fit every cell's rank-`rank` slice of `weight` (`q × p`) on the bank rows (`n × p`)
/// under a given partition.
pub fn fit_slice_atlas(
    bank: ArrayView2<'_, f64>,
    weight: ArrayView2<'_, f64>,
    centres: ArrayView2<'_, f64>,
    assignment: &[usize],
    rank: usize,
) -> Result<SliceAtlas, ChartSliceError> {
    let (n, p) = bank.dim();
    let (q, wp) = weight.dim();
    if n == 0 {
        return Err(ChartSliceError::EmptyBank);
    }
    if wp != p {
        return Err(ChartSliceError::WidthMismatch { bank: p, weight: wp });
    }
    if rank > q {
        return Err(ChartSliceError::RankTooLarge { rank, output_width: q });
    }
    let charts = centres.nrows();
    if charts == 0 {
        return Err(ChartSliceError::NoCharts);
    }
    if assignment.len() != n {
        return Err(ChartSliceError::InvalidAssignment { index: assignment.len().min(n) });
    }
    if let Some(index) = assignment.iter().position(|&c| c >= charts) {
        return Err(ChartSliceError::InvalidAssignment { index });
    }
    check_finite("bank", bank)?;
    check_finite("weight", weight)?;
    // W⁺ on the singular values resolved from zero by the backward band.
    let (u, sigma, vt) = weight.svd(true, true)?;
    let (u, vt) = (u.expect("requested"), vt.expect("requested"));
    let band = (q.max(p) as f64) * f64::EPSILON * sigma.iter().copied().fold(0.0, f64::max);
    let resolved = sigma.iter().take_while(|&&s| s > band).count();
    let u_r = u.slice(s![.., ..resolved]).to_owned(); // q × r
    let mut pinv_t = vt.slice(s![..resolved, ..]).to_owned(); // r × p, rows scaled by 1/σ
    for (mut row, s) in pinv_t.outer_iter_mut().zip(sigma.iter()) {
        row /= *s;
    }
    let mut out = Vec::with_capacity(charts);
    for c in 0..charts {
        let rows: Vec<usize> = (0..n).filter(|&i| assignment[i] == c).collect();
        let members = rows.len();
        let mean = if members > 0 {
            let mut m = Array1::<f64>::zeros(p);
            for &i in &rows {
                m += &bank.row(i);
            }
            m / members as f64
        } else {
            centres.row(c).to_owned()
        };
        let mut y = Array2::<f64>::zeros((members, q));
        for (k, &i) in rows.iter().enumerate() {
            let centred = &bank.row(i) - &mean;
            y.row_mut(k).assign(&weight.dot(&centred));
        }
        let (singular, kept) = if members == 0 {
            (Array1::zeros(0), Array2::<f64>::zeros((0, q)))
        } else {
            let (_, sv, yvt) = y.svd(false, true)?;
            let yvt = yvt.expect("requested");
            let d = rank.min(yvt.nrows());
            (sv, yvt.slice(s![..d, ..]).to_owned()) // d × q, orthonormal output directions
        };
        let d = kept.nrows();
        // Range check: O_c must lie in the resolved range of W.
        let coords = kept.dot(&u_r); // d × r
        let back = coords.dot(&u_r.t()); // d × q
        let leakage = (&kept - &back).iter().map(|v| v * v).sum::<f64>().sqrt();
        let range_band = (q as f64) * f64::EPSILON * (d.max(1) as f64).sqrt() * 8.0;
        if leakage > range_band && d > 0 {
            return Err(ChartSliceError::OutsideResolvedRange { chart: c, leakage, band: range_band });
        }
        let read = kept.dot(&weight); // d × p
        // write rows: (W⁺ O)ᵀ = Oᵀ (W⁺)ᵀ = Oᵀ U_r diag(1/σ) V_rᵀ = coords · pinv_t
        let write = coords.dot(&pinv_t); // d × p
        let mut padded = Array1::<f64>::zeros(singular.len());
        padded.assign(&singular);
        out.push(ChartSlice { mean, read, write, singular: padded, members });
    }
    Ok(SliceAtlas { centres: centres.to_owned(), charts: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn gaussian(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || {
            let u: f64 = rng.random_range(1e-12..1.0);
            let v: f64 = rng.random_range(0.0..1.0);
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
    }

    fn output_residual(atlas: &SliceAtlas, bank: ArrayView2<'_, f64>, weight: ArrayView2<'_, f64>) -> f64 {
        bank.outer_iter()
            .map(|z| {
                let e = weight.dot(&(&z - &atlas.restrict(z)));
                e.dot(&e)
            })
            .sum()
    }

    /// Rows on two separated affine 2-planes: two cells, rank 2, and the slices carry
    /// every row's output exactly.
    #[test]
    fn planted_planes_are_carried_exactly() {
        let mut rng = StdRng::seed_from_u64(2951);
        let p = 6;
        let weight = gaussian(&mut rng, 4, p);
        let frames = [gaussian(&mut rng, 2, p), gaussian(&mut rng, 2, p)];
        let offsets = [Array1::from_elem(p, 40.0), Array1::from_elem(p, -40.0)];
        let mut bank = Array2::<f64>::zeros((200, p));
        for i in 0..200 {
            let c = i % 2;
            let coef = gaussian(&mut rng, 1, 2);
            let row = &offsets[c] + &coef.row(0).dot(&frames[c]);
            bank.row_mut(i).assign(&row);
        }
        let (centres, assign) = lloyd_cells(bank.view(), 2).unwrap();
        assert!(assign.iter().enumerate().all(|(i, &c)| c == assign[i % 2]));
        assert_ne!(assign[0], assign[1]);
        let atlas = fit_slice_atlas(bank.view(), weight.view(), centres.view(), &assign, 2).unwrap();
        let scale: f64 = bank.outer_iter().map(|z| weight.dot(&z).dot(&weight.dot(&z))).sum();
        let direct = output_residual(&atlas, bank.view(), weight.view());
        assert!(direct <= 1e-20 * scale, "direct residual {direct:e} vs scale {scale:e}");
        assert!(atlas.bank_residual() <= 1e-20 * scale);
    }

    /// The reported residual is the directly measured output error, and no other
    /// rank-d slice of the same cell does better (Eckart–Young), including the one
    /// built from the input's own principal axes.
    #[test]
    fn slice_residual_is_exact_and_minimal() {
        let mut rng = StdRng::seed_from_u64(3);
        let (n, p, q, d) = (300, 9, 5, 2);
        let weight = gaussian(&mut rng, q, p);
        let mixing = gaussian(&mut rng, p, p);
        let bank = gaussian(&mut rng, n, p).dot(&mixing);
        let assign = vec![0usize; n];
        let centres = bank.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
        let atlas = fit_slice_atlas(bank.view(), weight.view(), centres.view(), &assign, d).unwrap();
        let direct = output_residual(&atlas, bank.view(), weight.view());
        let reported = atlas.bank_residual();
        assert!((direct - reported).abs() <= 1e-10 * atlas.charts()[0].bank_energy());
        // input-PCA slice of the same rank
        let mean = centres.row(0).to_owned();
        let centred = &bank - &mean;
        let (_, _, vt) = centred.svd(false, true).unwrap();
        let frame = vt.unwrap().slice(s![..d, ..]).to_owned();
        let pca: f64 = bank
            .outer_iter()
            .map(|z| {
                let c = &z - &mean;
                let zhat = &mean + &frame.t().dot(&frame.dot(&c));
                let e = weight.dot(&(&z - &zhat));
                e.dot(&e)
            })
            .sum();
        assert!(reported <= pca * (1.0 + 1e-12), "slice {reported} vs input PCA {pca}");
        // random rank-d oblique slices never beat it
        for _ in 0..50 {
            let r = gaussian(&mut rng, d, p);
            let a = gaussian(&mut rng, d, p);
            let other: f64 = bank
                .outer_iter()
                .map(|z| {
                    let c = &z - &mean;
                    let zhat = &mean + &a.t().dot(&r.dot(&c));
                    let e = weight.dot(&(&z - &zhat));
                    e.dot(&e)
                })
                .sum();
            assert!(reported <= other * (1.0 + 1e-12));
        }
    }

    /// Huge input spread in the weight's null space is ignored by the slice and
    /// captured by input PCA, which then misses everything the weight reads.
    #[test]
    fn nullspace_variance_is_ignored() {
        let mut rng = StdRng::seed_from_u64(11);
        let (n, p) = (400, 5);
        let mut weight = Array2::<f64>::zeros((2, p));
        weight[[0, 0]] = 1.0;
        weight[[1, 1]] = 1.0;
        let mut bank = gaussian(&mut rng, n, p);
        for mut row in bank.outer_iter_mut() {
            row[0] *= 1.0;
            row[1] *= 0.1;
            row[2] *= 100.0;
            row[3] *= 100.0;
            row[4] *= 100.0;
        }
        let assign = vec![0usize; n];
        let centres = bank.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));
        let atlas = fit_slice_atlas(bank.view(), weight.view(), centres.view(), &assign, 1).unwrap();
        let energy = atlas.charts()[0].bank_energy();
        let kept = energy - atlas.bank_residual();
        assert!(kept > 0.95 * energy, "slice keeps {kept} of {energy}");
        let mean = centres.row(0).to_owned();
        let (_, _, vt) = (&bank - &mean).svd(false, true).unwrap();
        let frame = vt.unwrap().slice(s![..1, ..]).to_owned();
        let pca: f64 = bank
            .outer_iter()
            .map(|z| {
                let c = &z - &mean;
                let zhat = &mean + &frame.t().dot(&frame.dot(&c));
                let e = weight.dot(&(&z - &zhat));
                e.dot(&e)
            })
            .sum();
        assert!(pca > 0.95 * energy, "input PCA misses {pca} of {energy}");
    }

    /// Lloyd's iteration stops at a fixed point: a second run from its own output
    /// partition moves nothing.
    #[test]
    fn lloyd_stops_at_a_fixed_point() {
        let mut rng = StdRng::seed_from_u64(7);
        let bank = gaussian(&mut rng, 500, 4);
        let (centres, assign) = lloyd_cells(bank.view(), 8).unwrap();
        let again: Vec<usize> = bank.outer_iter().map(|r| nearest(centres.view(), r)).collect();
        assert_eq!(assign, again);
    }
}

//! Structured descriptions of a block's map (#2951): what a rank-k block costs when its readers and
//! writers are named in a chart the decoder already holds and its core by its own structure, so a
//! block that does one simple thing costs what it takes to say that thing, not what its rank says.
//!
//! A block is the map `W = uᵀ v` (`d_out × d_in`) of some of a site's columns. A description writes
//!
//! ```text
//! W ≈ P_S K Q_Tᵀ
//! ```
//!
//! with `P` (`d_out × q_out`) and `Q` (`d_in × q_in`) charts coder and decoder both hold before the
//! block is decoded, `S` and `T` subsets of their column groups and `K` the core. The charts:
//!
//! * **Identity.** The side's own coordinates, every column taken: the generic block.
//! * **Coordinates** ([`Chart::coordinates`]). The side's coordinates in the groups its interface
//!   declares (attention heads, rotary planes): a block living on a few groups names them.
//! * **Harmonic** ([`Chart::harmonic`]). A side whose values the decoder holds on rows carrying
//!   labels in `Z_p^m`: a site's reads on the declared input family (decoded upstream, run on the
//!   family the contract declares), or a token map such as the unembedding on a residual write.
//!   Group `f` is the pair of directions `X⁺ cos 2π⟨f, t⟩/p`, `X⁺ sin 2π⟨f, t⟩/p`, so a reader whose
//!   profile over the inputs is one character is two numbers, not `d`.
//! * **Frames** ([`Chart::frames`], [`Chart::frame`]). Already decoded blocks' sides: a block that
//!   reads what another writes names it and sends only its core.
//!
//! Cores: generic rank `r` (`r (s + t − r)` reals, `K = A Bᵀ` with `B` the identity on `r` pivot
//! rows), or linear in a few reals when the two sides' chosen groups pair one to one (the pairing
//! sent in `log₂ g!` bits): rotation-scaling `[[a, −b], [b, a]]` (or the reflection
//! `[[a, b], [b, −a]]`, a bit a plane) per pair of two-column groups, or diagonal (one real per
//! paired column).
//!
//! Every real is sent on a dyadic lattice (`2^-p`, [`super::precision`], one `p` per factor) in the
//! signed Elias δ code ([`super::codec::signed_delta_len_bits`], under which splitting a real into
//! two never shortens a message); the structure (charts, groups, rank, pivots, exponents) in the
//! prefix integer and enumerative codes. A description's total is its bits plus `n KL / ln 2` of
//! the error its decoded map makes; a structured family is taken only when it lowers that total.
//! Everything is computed on the block's factors, so a description costs `O(d² r)` on a `d`-wide
//! site.
//!
//! # Pricing the error
//!
//! The search proposes with the second-order price `n ½ tr(F ΔW C ΔWᵀ) / ln 2` ([`Metric`]; `C` the
//! reads' second moment, `F` the written value's metric, best the logit-space Gauss-Newton
//! [`logit_gauss_newton`], which unlike the sampled-label Fisher does not vanish where the network
//! is confident). What decides is the error itself ([`Geometry::describe_exact`], [`Exact`]). For a
//! finite change `δ` of the logits `z`, with `p` the network's output, `p_t ∝ p e^{tδ}`,
//! `f(t) = log E_p e^{tδ} − t E_p δ` and `R = max δ − min δ` the oscillation of the change,
//!
//! ```text
//! KL(p ‖ p_1) = f(1) = ∫₀¹ (1 − t) f''(t) dt,    f''(t) = Var_{p_t}(δ),
//! e^{−tR} ≤ p_t / p ≤ e^{tR}   ⇒   e^{−tR} Var_p(δ) ≤ Var_{p_t}(δ) ≤ e^{tR} Var_p(δ),
//! 2 c₋(R) Q ≤ KL ≤ 2 c₊(R) Q,   Q = ½ Var_p(δ),   c₊(R) = (e^R − 1 − R)/R²,   c₋(R) = (e^{−R} − 1 + R)/R²
//! ```
//!
//! (the upper bound also as `log(1 + 2 c₊(R) Q)`, from `E_p e^X ≤ 1 + Var_p(X) c₊(R)` for the
//! centred `X`). Both factors are ½ at `R = 0`, so the quadratic price is exact to first order in
//! `R` and fails exactly when the oscillation of the change is large, not because `p` is peaked.
//! Where a site's writes reach the logits only through affine nodes, `δ` is linear in the block and
//! the bracket prices it without a forward ([`Exact::certified`]); elsewhere the decoded block is
//! run.
//!
//! # Planes, words and the library
//!
//! On a cyclic task (`p31`, `a + b mod p`) the characters' planes `H_f` are the irreducible real
//! modules of the shift `t ↦ t + 1`, with projector `(P_f)_{ab} = (2/p) cos(2πf(a − b)/p)`: what the
//! translation law forces once the library is paid once. Charged per word instead, a Fourier
//! reader `one-hot → (cos, sin)` has a box-exact decomposition into one rank-one slice per input
//! (`p + 1` numbers on a word) against the plane's `2p`, so per-word pricing prefers input-specific
//! slices, and on `p31` (every frequency used on every input) the gated fit finds no planes.
//! Measured on `p31` (`n = 10⁶`, every point decoded): the structured whole-site description,
//! library paid once over the 961 inputs, is 99 bits per word; rank-one subcomponents all on, 268;
//! the gated fits, about 3,000, almost all of it their KL.

use super::codec::{fixed_index_len_bits, prefix_integer_len_bits, signed_delta_len_bits, subset_code_len_bits};
use super::dense::{eigh, solve, svd};
use gam_linalg::faer_ndarray::fast_abt;
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;

/// The second-order price of a block's error: `n ½ tr(F ΔW C ΔWᵀ) / ln 2` bits, `C` the second
/// moment of the site's reads (the masked program reads them uncentred), `F` the Fisher of its
/// written value, `n` the observations the block's error is paid on.
#[derive(Clone, Debug)]
pub struct Metric {
    pub moment: Array2<f64>,
    pub fisher: Array2<f64>,
    pub observations: f64,
}

impl Metric {
    /// The metric of a site from its measured statistics ([`super::masked::site_statistics`]).
    pub fn of(site: &super::pieces::Site, observations: f64) -> Self {
        Self { moment: symmetric(&site.second_moment), fisher: symmetric(&site.fisher), observations }
    }

    /// Bits per unit of the whitened squared error `tr(F ΔW C ΔWᵀ)`.
    fn scale(&self) -> f64 {
        0.5 * self.observations / std::f64::consts::LN_2
    }
}

/// A run of a chart's columns described together: on a harmonic chart one character's cosine and
/// sine, `mode` its frequency vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub start: usize,
    pub width: usize,
    pub mode: Vec<usize>,
}

/// A decodable basis of one side of a site (module note).
#[derive(Clone, Debug)]
pub struct Chart {
    pub name: String,
    /// `d × q`: chart coordinates to the side's coordinates.
    pub basis: Array2<f64>,
    pub groups: Vec<Group>,
    /// Whether a description names a subset of the groups (else it takes them all).
    pub subsets: bool,
}

impl Chart {
    /// The side's own coordinates.
    pub fn identity(d: usize) -> Self {
        Self { name: "identity".to_string(), basis: Array2::eye(d), groups: vec![Group { start: 0, width: d, mode: Vec::new() }], subsets: false }
    }

    /// The columns of an already decoded frame (`d × k`), taken together.
    pub fn frame(name: &str, columns: Array2<f64>) -> Self {
        let width = columns.ncols();
        Self { name: name.to_string(), basis: columns, groups: vec![Group { start: 0, width, mode: Vec::new() }], subsets: false }
    }

    /// The side's own coordinates in the groups its interface declares (attention heads, rotary
    /// planes), a description naming the groups its block lives on: `groups` lists each group's
    /// coordinates, every coordinate in at most one.
    pub fn coordinates(name: &str, d: usize, groups: &[Vec<usize>]) -> Result<Self, String> {
        let width: usize = groups.iter().map(Vec::len).sum();
        let mut basis = Array2::<f64>::zeros((d, width));
        let mut seen = vec![false; d];
        let mut out = Vec::new();
        let mut start = 0;
        for (i, group) in groups.iter().enumerate() {
            for (c, &j) in group.iter().enumerate() {
                if j >= d || seen[j] {
                    return Err(format!("coordinate {j} out of 0..{d} or in two groups"));
                }
                seen[j] = true;
                basis[[j, start + c]] = 1.0;
            }
            out.push(Group { start, width: group.len(), mode: vec![i] });
            start += group.len();
        }
        Ok(Self { name: name.to_string(), basis, groups: out, subsets: true })
    }

    /// The frames of already decoded blocks (`columns`, `d × C`, block `i` its next `widths[i]`
    /// columns), a description naming the blocks whose frames it reads or writes.
    pub fn frames(name: &str, columns: Array2<f64>, widths: &[usize]) -> Result<Self, String> {
        if widths.iter().sum::<usize>() != columns.ncols() {
            return Err(format!("{} frame columns in blocks of {widths:?}", columns.ncols()));
        }
        let mut start = 0;
        let groups = widths
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let g = Group { start, width: *w, mode: vec![i] };
                start += w;
                g
            })
            .collect();
        Ok(Self { name: name.to_string(), basis: columns, groups, subsets: true })
    }

    /// The harmonic chart of a side whose values the decoder holds on labelled rows: `values`
    /// (`rows × d`) the side's value on each row, `labels` (`rows × m`, entries in `0..period`) the
    /// row's labels (the operands of an input, or a token id when `values` is a token map). Group
    /// `f ∈ Z_period^m` (one of each pair `±f`) spans the directions whose profiles over the rows
    /// are `cos 2π⟨f, t⟩/period` and `sin 2π⟨f, t⟩/period` (the cosine alone where `f = −f`): the
    /// least-squares directions `X⁺ h`, `X⁺` the pseudo-inverse of `values` beyond its rounding band,
    /// each scaled to a unit mean-square profile `‖X b‖² = rows`.
    pub fn harmonic(name: &str, values: ArrayView2<'_, f64>, labels: ArrayView2<'_, usize>, period: usize) -> Result<Self, String> {
        let (rows, d) = values.dim();
        let m = labels.ncols();
        if labels.nrows() != rows || period < 2 || m == 0 {
            return Err(format!("a harmonic chart needs labels on every row ({} for {rows}) and a period of at least two ({period})", labels.nrows()));
        }
        if labels.iter().any(|l| *l >= period) {
            return Err(format!("a label is outside 0..{period}"));
        }
        let decomposed = svd(values, false).map_err(|e| format!("{e:?}"))?;
        let kept: Vec<usize> = (0..decomposed.singular_values.len()).filter(|i| decomposed.singular_values[*i] > decomposed.band).collect();
        // X⁺ = V Σ⁺ Uᵀ over the resolved singular values, applied to each profile as V (Σ⁺ (Uᵀ h)).
        let u = decomposed.u.select(Axis(1), &kept);
        let v = decomposed.vt.select(Axis(0), &kept).reversed_axes();
        let inverse = Array1::from_iter(kept.iter().map(|i| 1.0 / decomposed.singular_values[*i]));
        let total = period.pow(m as u32);
        let unrank = |mut r: usize| {
            let mut f = vec![0; m];
            for x in f.iter_mut().rev() {
                *x = r % period;
                r /= period;
            }
            f
        };
        let rank = |f: &[usize]| f.iter().fold(0, |r, x| r * period + x);
        let mut profiles: Vec<Array1<f64>> = Vec::new();
        let mut groups = Vec::new();
        for r in 0..total {
            let f = unrank(r);
            let conjugate: Vec<usize> = f.iter().map(|x| (period - x) % period).collect();
            let c = rank(&conjugate);
            if c < r {
                continue;
            }
            let angle = |row: usize| {
                let phase = f.iter().zip(labels.row(row)).map(|(a, b)| a * b).sum::<usize>() % period;
                2.0 * std::f64::consts::PI * phase as f64 / period as f64
            };
            let start = profiles.len();
            profiles.push(Array1::from_shape_fn(rows, |row| angle(row).cos()));
            if c != r {
                profiles.push(Array1::from_shape_fn(rows, |row| angle(row).sin()));
            }
            groups.push(Group { start, width: profiles.len() - start, mode: f });
        }
        // Each direction scaled to a unit mean-square profile on the rows: a character the values
        // barely resolve would otherwise come back as a reader of enormous norm, whose products
        // with the metric cancel in floating point.
        let mut basis = Array2::<f64>::zeros((d, profiles.len()));
        for (j, h) in profiles.iter().enumerate() {
            let direction = v.dot(&(&u.t().dot(h) * &inverse));
            let profile = values.dot(&direction);
            let rms = (profile.dot(&profile) / rows as f64).sqrt();
            if rms > 0.0 {
                basis.column_mut(j).assign(&(direction / rms));
            }
        }
        Ok(Self { name: name.to_string(), basis, groups, subsets: true })
    }

    /// The basis columns of `groups`, in their order.
    fn indices(&self, groups: &[usize]) -> Vec<usize> {
        groups.iter().flat_map(|g| self.groups[*g].start..self.groups[*g].start + self.groups[*g].width).collect()
    }

    /// Bits naming `chosen` of the groups (nothing when the chart takes them all).
    fn subset_bits(&self, chosen: usize) -> Result<f64, String> {
        if !self.subsets {
            return Ok(0.0);
        }
        Ok(subset_code_len_bits(self.groups.len(), chosen).map_err(|e| e.to_string())? as f64)
    }
}

/// The core of a description.
#[derive(Clone, Debug, PartialEq)]
pub enum Core {
    /// Generic rank `r`, `K = A Bᵀ` with `B` the identity on its pivot rows.
    Generic { rank: usize },
    /// Per pair of a writer and a reader group (paired in the order listed, the pairing sent in
    /// `log₂ g!` bits), `[[a, −b], [b, a]]` on two-column groups, or `[[a, b], [b, −a]]` where the
    /// pair's flag is set; a one-column pair is one real.
    Rotation { reflections: Vec<bool> },
    /// One real per paired column.
    Diagonal,
}

/// One block's chosen description.
#[derive(Clone, Debug)]
pub struct Description {
    /// Writer and reader chart names and the modes of their chosen groups.
    pub writer: (String, Vec<Vec<usize>>),
    pub reader: (String, Vec<Vec<usize>>),
    pub core: Core,
    /// The reals it sends.
    pub reals: usize,
    pub structure_bits: f64,
    pub real_bits: f64,
    /// The second-order KL bits of the decoded map's error against the block.
    pub kl_bits: f64,
    /// The decoded block as factors, `W = uᵀ v`: `u` is `c × d_out`, `v` is `c × d_in`.
    pub u: Array2<f64>,
    pub v: Array2<f64>,
    /// What [`Geometry::recode`] needs to send another block in the same family at the same
    /// precision; `None` for the empty description.
    pub choice: Option<Choice>,
}

/// A description's family and precision: the charts and their groups and the lattice exponents
/// (the core's kind is the description's).
#[derive(Clone, Debug)]
pub struct Choice {
    writer: usize,
    writer_groups: Vec<usize>,
    reader: usize,
    reader_groups: Vec<usize>,
    exponents: Vec<i32>,
}

impl Choice {
    /// The lattice exponent its reader side was sent at (the generic core's second factor, a linear
    /// core's one).
    pub fn reader_exponent(&self) -> Option<i32> {
        self.exponents.last().copied()
    }
}


impl Description {
    /// The description's own bits.
    pub fn bits(&self) -> f64 {
        self.structure_bits + self.real_bits
    }

    /// Its bits plus the price of its error.
    pub fn total(&self) -> f64 {
        self.bits() + self.kl_bits
    }
}

fn symmetric(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) * 0.5
}

/// `a b` through faer, so inside a site's per-subcomponent pricing (run in parallel under
/// `gam_linalg::faer_ndarray::with_nested_parallel`) it stays on its own thread.
fn mm<A: ndarray::Data<Elem = f64>, B: ndarray::Data<Elem = f64>>(a: &ndarray::ArrayBase<A, ndarray::Ix2>, b: &ndarray::ArrayBase<B, ndarray::Ix2>) -> Array2<f64> {
    gam_linalg::faer_ndarray::fast_ab(a, b)
}

/// The pseudo-inverse and the pseudo-inverse root of a symmetric positive semidefinite matrix,
/// over its eigenvalues beyond the decomposition's band.
fn inverses(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let d = eigh(symmetric(m).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    // `Q Λ⁻¹ Qᵀ` and `Q Λ^{-1/2} Qᵀ` over the eigenvalues beyond the band, as two products of the
    // scaled eigenvectors (the dropped ones scaled to zero), each mirrored to exact symmetry.
    let (mut inverse_factor, mut root_factor) = (d.vectors.clone(), d.vectors.clone());
    for (i, l) in d.values.iter().enumerate() {
        let (a, b) = if *l > d.band { (1.0 / l, 1.0 / l.sqrt()) } else { (0.0, 0.0) };
        inverse_factor.column_mut(i).mapv_inplace(|v| v * a);
        root_factor.column_mut(i).mapv_inplace(|v| v * b);
    }
    Ok((symmetric(&fast_abt(&inverse_factor, &d.vectors)), symmetric(&fast_abt(&root_factor, &d.vectors))))
}

/// Bits of a real sent as the signed integer `i` (the signed Elias δ code).
fn integer_bits(i: i64) -> f64 {
    signed_delta_len_bits(i).map(|b| b as f64).unwrap_or(f64::INFINITY)
}

/// `x` on the lattice `2^-p` and the bits of its integers, or `None` when an integer leaves the
/// exactly representable range.
fn quantize(x: &Array2<f64>, p: i32) -> Option<(Array2<f64>, f64)> {
    let scale = 2f64.powi(p);
    let mut bits = 0.0;
    let mut out = Array2::<f64>::zeros(x.dim());
    for (o, v) in out.iter_mut().zip(x.iter()) {
        let k = (v * scale).round();
        if !(k.abs() < 2f64.powi(52)) {
            return None;
        }
        bits += integer_bits(k as i64);
        *o = k / scale;
    }
    Some((out, bits))
}

/// The exponents worth scanning for `x`: from the step at which every entry rounds to zero to the
/// step at which the largest entry spends the whole mantissa.
fn exponents(x: &Array2<f64>) -> std::ops::RangeInclusive<i32> {
    let largest = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if largest == 0.0 || !largest.is_finite() {
        return 0..=0;
    }
    let top = -(largest.log2().ceil() as i32) - 1;
    top..=top + 52
}

/// Bits of a lattice exponent.
fn exponent_bits(p: i32) -> f64 {
    integer_bits(i64::from(p))
}

/// What a block's description is fitted against on one choice of charts: the block's
/// `H = Pᵀ F W C Q = H_l H_rᵀ`, `G_p = Pᵀ F P`, `G_q = Qᵀ C Q` and their pseudo-inverse roots, so a
/// core `K` leaves the whitened squared error `w2 − 2⟨K, H⟩ + tr(G_p K G_q Kᵀ)`.
/// On an identity side the basis is `None` and the Gram and root are the metric's own, borrowed.
struct Sides<'a> {
    p: Option<Array2<f64>>,
    q: Option<Array2<f64>>,
    hl: Array2<f64>,
    hr: Array2<f64>,
    h: Array2<f64>,
    gp: Cow<'a, Array2<f64>>,
    gq: Cow<'a, Array2<f64>>,
    rp: Cow<'a, Array2<f64>>,
    rq: Cow<'a, Array2<f64>>,
}

impl Sides<'_> {
    /// The decoded factors `(P A)ᵀ`, `(Q B)ᵀ` of a core `A Bᵀ` in chart coordinates.
    fn decoded(&self, a: &Array2<f64>, b: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let side = |basis: &Option<Array2<f64>>, f: &Array2<f64>| match basis {
            Some(x) => mm(x, f).reversed_axes(),
            None => f.t().to_owned(),
        };
        (side(&self.p, a), side(&self.q, b))
    }
}

/// The block's factored products every choice of charts reuses: `F uᵀ`, `C vᵀ`, `u F uᵀ`,
/// `v C vᵀ` and `w2 = tr(F W C Wᵀ) = tr[(u F uᵀ)(v C vᵀ)]`.
struct Block<'a> {
    metric: &'a Metric,
    /// Bits per unit of whitened squared error: the metric's, times the calibration.
    scale: f64,
    fu: Array2<f64>,
    cv: Array2<f64>,
    gu: Array2<f64>,
    gv: Array2<f64>,
    w2: f64,
}

impl<'a> Block<'a> {
    fn new(u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>, metric: &'a Metric, calibration: f64) -> Self {
        let fu = mm(&metric.fisher, &u.t());
        let cv = mm(&metric.moment, &v.t());
        let gu = symmetric(&mm(&u, &fu));
        let gv = symmetric(&mm(&v, &cv));
        let w2 = (&gu * &gv).sum();
        Self { metric, scale: metric.scale() * calibration, fu, cv, gu, gv, w2 }
    }

    /// The sides of a choice of groups of two prepared charts.
    fn sides<'p>(&self, writer: &'p Prepared, wg: &[usize], reader: &'p Prepared, rg: &[usize]) -> Result<Sides<'p>, String>
    where
        'a: 'p,
    {
        let side = |chart: &'p Prepared, groups: &[usize], factor: &Array2<f64>, metric: &'p Array2<f64>| -> Result<_, String> {
            if chart.identity {
                let root = chart.root.as_ref().ok_or("an identity chart without its root")?;
                Ok((None, factor.clone(), Cow::Borrowed(metric), Cow::Borrowed(root)))
            } else {
                let at = chart.chart.indices(groups);
                let x = chart.chart.basis.select(Axis(1), &at);
                let g = chart.gram(&at);
                let root = inverses(&g)?.1;
                let h = mm(&x.t(), factor);
                Ok((Some(x), h, Cow::Owned(g), Cow::Owned(root)))
            }
        };
        let (p, hl, gp, rp) = side(writer, wg, &self.fu, &self.metric.fisher)?;
        let (q, hr, gq, rq) = side(reader, rg, &self.cv, &self.metric.moment)?;
        // The dense `H` only for the linear cores, which pair two charts' groups; a pair with an
        // identity side codes generic cores only, on `H`'s factors.
        let h = if writer.identity || reader.identity { Array2::zeros((0, 0)) } else { mm(&hl, &hr.t()) };
        Ok(Sides { p, q, hl, hr, h, gp, gq, rp, rq })
    }

    /// The KL bits of the factored core `K = A Bᵀ`.
    fn error_factored(&self, sides: &Sides, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let cross = (mm(&a.t(), &sides.hl) * mm(&b.t(), &sides.hr)).sum();
        let quad = (mm(&mm(&a.t(), &*sides.gp), a) * mm(&mm(&b.t(), &*sides.gq), b)).sum();
        (self.w2 - 2.0 * cross + quad).max(0.0) * self.scale
    }
}

/// The best exponent of a scan: `cost(p)` is `(bits, KL bits)`, `None` when `p` is out of range.
/// The lattice bits grow as it refines and the error they leave falls, so their sum is searched as
/// unimodal in `p` (ternary search, each exponent costed once).
fn scan(range: std::ops::RangeInclusive<i32>, mut cost: impl FnMut(i32) -> Option<(f64, f64)>) -> Option<(i32, f64)> {
    let (start, end) = (*range.start(), *range.end());
    if end < start {
        return None;
    }
    let len = (end - start + 1) as usize;
    let mut total = |n: usize| -> Result<f64, String> {
        Ok(cost(start + n as i32).map_or(f64::INFINITY, |(b, k)| b + k)).map(|t| if t.is_nan() { f64::INFINITY } else { t })
    };
    let mut values: Vec<Option<f64>> = vec![None; len];
    let mut memo = |n: usize| -> f64 {
        if let Some(v) = values[n] {
            return v;
        }
        let v = total(n).unwrap_or(f64::INFINITY);
        values[n] = Some(v);
        v
    };
    let n = minimize(len, &mut |n| Ok(memo(n))).ok()??;
    Some((start + n as i32, memo(n)))
}

/// The exponent minimizing `exact` near the minimum of `surrogate` over `range`: the surrogate's scan
/// ([`scan`]), then exact steps from its minimum toward the cheaper neighbour while it is cheaper,
/// each exponent costed exactly once; where the exact cost cannot price the surrogate's minimum, the
/// exact cost's own scan.
fn refine(range: std::ops::RangeInclusive<i32>, surrogate: impl FnMut(i32) -> Option<(f64, f64)>, mut exact: impl FnMut(i32) -> Option<(f64, f64)>) -> Option<i32> {
    let (start, end) = (*range.start(), *range.end());
    let guess = scan(range.clone(), surrogate).map(|(p, _)| p);
    let mut memo: HashMap<i32, f64> = HashMap::new();
    let mut total = |p: i32| -> f64 {
        if let Some(v) = memo.get(&p) {
            return *v;
        }
        let v = exact(p).map_or(f64::INFINITY, |(b, k)| b + k);
        let v = if v.is_nan() { f64::INFINITY } else { v };
        memo.insert(p, v);
        v
    };
    let Some(mut at) = guess.filter(|p| total(*p).is_finite()) else {
        return scan(range, |p| Some((total(p), 0.0))).map(|(p, _)| p);
    };
    let inside = |p: i32| (start..=end).contains(&p);
    let step = [-1, 1].into_iter().filter(|s| inside(at + s)).min_by(|x, y| total(at + x).total_cmp(&total(at + y)));
    if let Some(step) = step {
        while inside(at + step) && total(at + step) < total(at) {
            at += step;
        }
    }
    Some(at)
}

/// Up to `rank` pivot columns of `k` by greedy residual norm (maximum volume, column by column).
fn pivots(k: &Array2<f64>, rank: usize) -> Vec<usize> {
    let mut residual = k.clone();
    let mut chosen = Vec::new();
    for _ in 0..rank.min(k.ncols()) {
        let norms: Vec<f64> = residual.columns().into_iter().map(|c| c.dot(&c)).collect();
        let Some((j, n)) = norms.iter().enumerate().filter(|(j, _)| !chosen.contains(j)).max_by(|a, b| a.1.total_cmp(b.1)) else { break };
        if *n <= 0.0 {
            break;
        }
        let e = residual.column(j).to_owned() / n.sqrt();
        let projection = e.view().insert_axis(Axis(1)).dot(&e.view().insert_axis(Axis(0)).dot(&residual).view());
        residual -= &projection;
        chosen.push(j);
    }
    chosen
}

/// A fitted and coded core, with its decoded factors in chart coordinates: `K̃ = A Bᵀ`.
struct Coded {
    a: Array2<f64>,
    b: Array2<f64>,
    reals: usize,
    /// The lattice integers' bits, and the rest (exponents, rank, pivots, flags).
    real_bits: f64,
    structure_bits: f64,
    kl: f64,
    /// The lattice exponents its reals were sent at.
    exponents: Vec<i32>,
}

impl Coded {
    fn total(&self) -> f64 {
        self.real_bits + self.structure_bits + self.kl
    }
}

/// The generic cores on `sides` of the ranks a search over `1..=max_rank` visits, each the metric's best (the whitened
/// `M = G_p^{+1/2} H G_q^{+1/2} = A Bᵀ` truncated, its singular pairs from the `r × r` core
/// `G_a^{1/2} G_b G_a^{1/2}` of `A`'s and `B`'s Grams) and coded in the pivot chart, its two
/// exponents the minimum of bits plus error by coordinate descent.
fn generic_cores(block: &Block<'_>, sides: &Sides, max_rank: usize) -> Result<Vec<Coded>, String> {
    let a = mm(&*sides.rp, &sides.hl);
    let b = mm(&*sides.rq, &sides.hr);
    let (half, inverse_half) = roots(&mm(&a.t(), &a))?;
    let core = symmetric(&mm(&mm(&half, &mm(&b.t(), &b)), &half));
    let decomposed = eigh(core.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut order: Vec<usize> = (0..decomposed.values.len()).filter(|i| decomposed.values[*i] > decomposed.band).collect();
    order.sort_by(|x, y| decomposed.values[*y].total_cmp(&decomposed.values[*x]));
    // m = Σ (A G_a^{-1/2} z)(B G_a^{1/2} z)ᵀ over the core's eigenvectors z.
    let left = mm(&*sides.rp, &mm(&a, &inverse_half));
    let right = mm(&*sides.rq, &mm(&b, &half));
    // The rank by ternary search on the total (bits grow with it, the error left falls).
    let ranks = max_rank.min(order.len());
    let mut coded: Vec<Option<Option<Coded>>> = (0..ranks).map(|_| None).collect();
    let mut failure = None;
    minimize(ranks, &mut |n| {
        if coded[n].is_none() {
            let z = decomposed.vectors.select(Axis(1), &order[..n + 1]);
            match generic_core(block, sides, &mm(&left, &z), &mm(&right, &z), None) {
                Ok(c) => coded[n] = Some(c),
                Err(e) => {
                    failure = Some(e);
                    coded[n] = Some(None);
                }
            }
        }
        Ok(coded[n].as_ref().and_then(|c| c.as_ref()).map_or(f64::INFINITY, Coded::total))
    })?;
    if let Some(e) = failure {
        return Err(e);
    }
    Ok(coded.into_iter().flatten().flatten().collect())
}

/// The generic core of rank `rank` on `sides` coded at the exponents `fixed` (no search).
fn generic_core_at(block: &Block<'_>, sides: &Sides, rank: usize, fixed: &[i32]) -> Result<Option<Coded>, String> {
    let a = mm(&*sides.rp, &sides.hl);
    let b = mm(&*sides.rq, &sides.hr);
    let (half, inverse_half) = roots(&mm(&a.t(), &a))?;
    let core = symmetric(&mm(&mm(&half, &mm(&b.t(), &b)), &half));
    let decomposed = eigh(core.view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut order: Vec<usize> = (0..decomposed.values.len()).filter(|i| decomposed.values[*i] > decomposed.band).collect();
    order.sort_by(|x, y| decomposed.values[*y].total_cmp(&decomposed.values[*x]));
    if order.len() < rank || rank == 0 {
        return Ok(None);
    }
    let z = decomposed.vectors.select(Axis(1), &order[..rank]);
    let left = mm(&mm(&*sides.rp, &mm(&a, &inverse_half)), &z);
    let right = mm(&mm(&*sides.rq, &mm(&b, &half)), &z);
    generic_core(block, sides, &left, &right, Some(fixed))
}

/// The core `K = K_l K_rᵀ` (rank `k` = their width) coded in the pivot chart `K = A Bᵀ`,
/// `A = K_{:,π} = K_l K_r[π]ᵀ` and `B = K_r K_r[π]⁻¹`, the identity on the pivot rows `π`.
fn generic_core(block: &Block<'_>, sides: &Sides, kl: &Array2<f64>, kr: &Array2<f64>, fixed: Option<&[i32]>) -> Result<Option<Coded>, String> {
    let (s_, t_) = (kl.nrows(), kr.nrows());
    let pivot = pivots(&kr.t().to_owned(), kr.ncols());
    let rank = pivot.len();
    if rank < kr.ncols() || rank == 0 {
        return Ok(None);
    }
    let m = kr.select(Axis(0), &pivot);
    let a = mm(kl, &m.t());
    let Ok(bt) = solve(m.t(), kr.t()) else { return Ok(None) };
    let mut b = bt.reversed_axes();
    for (i, &j) in pivot.iter().enumerate() {
        b.row_mut(j).fill(0.0);
        b[[j, i]] = 1.0;
    }
    let free: Vec<usize> = (0..t_).filter(|j| !pivot.contains(j)).collect();
    let b_free = b.select(Axis(0), &free);
    let rebuild = |b_free: &Array2<f64>| {
        let mut full = b.clone();
        for (i, &j) in free.iter().enumerate() {
            full.row_mut(j).assign(&b_free.row(i));
        }
        full
    };
    let cost = |pa: i32, pb: i32| -> Option<(f64, f64, Array2<f64>, Array2<f64>)> {
        let (qa, ba) = quantize(&a, pa)?;
        let (qb, bb) = if free.is_empty() { (b_free.clone(), 0.0) } else { quantize(&b_free, pb)? };
        let full = rebuild(&qb);
        let kl = block.error_factored(sides, &qa, &full);
        Some((ba + bb, kl, qa, full))
    };
    let exponent_cost = |pa: i32, pb: i32| exponent_bits(pa) + if free.is_empty() { 0.0 } else { exponent_bits(pb) };
    let scale = block.scale;
    // The scans' surrogate of a rounded factor's form: `x̃ = x + e` has
    // `x̃ᵀ G x̃ = xᵀGx + eᵀ(Gx) + (eᵀ(Gx))ᵀ + eᵀGe`, all `O(d r²)` once `Gx` is formed but the last,
    // which the surrogate takes on `G`'s diagonal (its mean under independent rounding errors). The
    // exact form decides, at the surrogate's minimum and the neighbours it steps to ([`refine`]).
    let (gpa, gqb) = (mm(&*sides.gp, &a), mm(&*sides.gq, &b));
    let (aga, bgb) = (mm(&a.t(), &gpa), mm(&b.t(), &gqb));
    let (dp, dq) = (sides.gp.diag().to_owned(), sides.gq.diag().to_owned());
    let surrogate = |x: &Array2<f64>, gx: &Array2<f64>, xgx: &Array2<f64>, diagonal: &Array1<f64>, rounded: &Array2<f64>| -> Array2<f64> {
        let e = rounded - x;
        let cross = mm(&e.t(), gx);
        let weighted = &e * &diagonal.view().insert_axis(Axis(1));
        xgx + &cross + &cross.t() + &mm(&weighted.t(), &e)
    };
    // While one factor is held, the error is a quadratic in the other: its products with the held
    // factor are formed once per scan, `w2 − 2 tr[(ÃᵀH_l)(H_rᵀB̃)ᵀ] + tr[(ÃᵀG_pÃ)(B̃ᵀG_qB̃)]`.
    let scan_a = |pb: i32| -> Option<i32> {
        let (qb, bb) = if free.is_empty() { (b_free.clone(), 0.0) } else { quantize(&b_free, pb)? };
        let full = rebuild(&qb);
        let (hb, gb) = (mm(&sides.hr.t(), &full), mm(&mm(&full.t(), &*sides.gq), &full));
        let error = |qa: &Array2<f64>, form: &Array2<f64>| {
            let cross = (mm(&qa.t(), &sides.hl) * &hb.t()).sum();
            (block.w2 - 2.0 * cross + (form * &gb).sum()).max(0.0) * scale
        };
        refine(
            exponents(&a),
            |p| {
                let (qa, ba) = quantize(&a, p)?;
                Some((ba + bb + exponent_cost(p, pb), error(&qa, &surrogate(&a, &gpa, &aga, &dp, &qa))))
            },
            |p| {
                let (qa, ba) = quantize(&a, p)?;
                Some((ba + bb + exponent_cost(p, pb), error(&qa, &mm(&mm(&qa.t(), &*sides.gp), &qa))))
            },
        )
    };
    let scan_b = |pa: i32| -> Option<i32> {
        let (qa, ba) = quantize(&a, pa)?;
        let (ha, ga) = (mm(&qa.t(), &sides.hl), mm(&mm(&qa.t(), &*sides.gp), &qa));
        let error = |full: &Array2<f64>, form: &Array2<f64>| {
            let cross = (&ha * &mm(&full.t(), &sides.hr)).sum();
            (block.w2 - 2.0 * cross + (&ga * form).sum()).max(0.0) * scale
        };
        refine(
            exponents(&b_free),
            |p| {
                let (qb, bb) = quantize(&b_free, p)?;
                let full = rebuild(&qb);
                Some((ba + bb + exponent_cost(pa, p), error(&full, &surrogate(&b, &gqb, &bgb, &dq, &full))))
            },
            |p| {
                let (qb, bb) = quantize(&b_free, p)?;
                let full = rebuild(&qb);
                Some((ba + bb + exponent_cost(pa, p), error(&full, &mm(&mm(&full.t(), &*sides.gq), &full))))
            },
        )
    };
    let (mut pa, mut pb) = match fixed {
        Some(&[pa, pb]) => (pa, pb),
        _ => (*exponents(&a).end(), *exponents(&b_free).end()),
    };
    for _ in 0..if fixed.is_some() { 0 } else { 2 } {
        if let Some(p) = scan_a(pb) {
            pa = p;
        }
        if !free.is_empty()
            && let Some(p) = scan_b(pa)
        {
            pb = p;
        }
    }
    let Some((bits, kl, qa, qb)) = cost(pa, pb) else { return Ok(None) };
    let structure = exponent_cost(pa, pb)
        + prefix_integer_len_bits(rank as u64).map_err(|e| e.to_string())? as f64
        + subset_code_len_bits(t_, rank).map_err(|e| e.to_string())? as f64;
    Ok(Some(Coded { a: qa, b: qb, reals: rank * (s_ + t_ - rank), real_bits: bits, structure_bits: structure, kl, exponents: vec![pa, pb] }))
}

/// One parameter of a linear core: entries `(i, j, c)` of `∂K/∂θ`.
type Placement = Vec<(usize, usize, f64)>;

/// The core `K = Σ θ_p E_p` on `sides`, fitted in the metric (its normal equations) and coded at one
/// exponent.
fn linear_core(block: &Block<'_>, sides: &Sides, placements: &[Placement], structure: f64, fixed: Option<i32>) -> Result<Option<Coded>, String> {
    let m = placements.len();
    if m == 0 {
        return Ok(None);
    }
    let mut gram = Array2::<f64>::zeros((m, m));
    let mut rhs = Array1::<f64>::zeros(m);
    for (p, ep) in placements.iter().enumerate() {
        rhs[p] = ep.iter().map(|(i, j, c)| c * sides.h[[*i, *j]]).sum();
        for (q, eq) in placements.iter().enumerate() {
            let mut g = 0.0;
            for (i, j, c) in ep {
                for (k, l, d) in eq {
                    g += c * d * sides.gp[[*i, *k]] * sides.gq[[*l, *j]];
                }
            }
            gram[[p, q]] = g;
        }
    }
    let (inverse, _) = inverses(&gram)?;
    let theta = inverse.dot(&rhs).insert_axis(Axis(1));
    // `K = Σ θ_p E_p` is linear in `θ`, so its error is `w2 − 2 θᵀ r + θᵀ G θ` on the placements'
    // Gram and right-hand side: exact, without forming `K`.
    let error = |q: &Array2<f64>| {
        let t = q.column(0);
        (block.w2 - 2.0 * t.dot(&rhs) + t.dot(&gram.dot(&t))).max(0.0) * block.scale
    };
    let (s_, t_) = sides.h.dim();
    let core = |theta: &Array2<f64>| {
        let mut k = Array2::<f64>::zeros((s_, t_));
        for (p, ep) in placements.iter().enumerate() {
            for (i, j, c) in ep {
                k[[*i, *j]] += c * theta[[p, 0]];
            }
        }
        k
    };
    let cost = |p: i32| -> Option<(f64, f64, Array2<f64>)> {
        let (q, bits) = quantize(&theta, p)?;
        Some((bits, error(&q), q))
    };
    let p = match fixed {
        Some(p) => p,
        None => {
            let Some((p, _)) = scan(exponents(&theta), |p| cost(p).map(|(b, k, _)| (b + exponent_bits(p), k))) else { return Ok(None) };
            p
        }
    };
    let Some((bits, kl, q)) = cost(p) else { return Ok(None) };
    let k = core(&q);
    // Factored as K̃ = K̃ · I.
    Ok(Some(Coded { a: k, b: Array2::eye(t_), reals: m, real_bits: bits, structure_bits: structure + exponent_bits(p), kl, exponents: vec![p] }))
}

/// The chosen groups of two charts paired one to one, each writer group with the reader group of
/// equal width it couples to most in the metric (`‖G_p^{-1/2} H G_q^{-1/2}‖²` on their block,
/// greedily), as `(writer offset, reader offset, width)`; `None` when they do not pair.
fn paired(writer: &Chart, w: &[usize], reader: &Chart, r: &[usize], sides: &Sides) -> Result<Option<Vec<(usize, usize, usize)>>, String> {
    if w.len() != r.len() {
        return Ok(None);
    }
    let offsets = |chart: &Chart, groups: &[usize]| {
        let mut at = 0;
        groups
            .iter()
            .map(|g| {
                let o = at;
                at += chart.groups[*g].width;
                (o, chart.groups[*g].width)
            })
            .collect::<Vec<_>>()
    };
    let (wo, ro) = (offsets(writer, w), offsets(reader, r));
    let roots = |g: &Array2<f64>, at: &[(usize, usize)]| -> Result<Vec<Array2<f64>>, String> {
        at.iter().map(|(o, n)| inverses(&g.slice(s![*o..o + n, *o..o + n]).to_owned()).map(|x| x.1)).collect()
    };
    let (rw, rr) = (roots(&*sides.gp, &wo)?, roots(&*sides.gq, &ro)?);
    let mut couplings = Vec::new();
    for (i, (o, n)) in wo.iter().enumerate() {
        for (j, (q, m)) in ro.iter().enumerate() {
            if n == m {
                let block = rw[i].dot(&sides.h.slice(s![*o..o + n, *q..q + m])).dot(&rr[j]);
                couplings.push(((&block * &block).sum(), i, j));
            }
        }
    }
    couplings.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut used_w, mut used_r) = (vec![false; wo.len()], vec![false; ro.len()]);
    let mut out = Vec::new();
    for (_, i, j) in couplings {
        if !used_w[i] && !used_r[j] {
            used_w[i] = true;
            used_r[j] = true;
            out.push((wo[i].0, ro[j].0, wo[i].1));
        }
    }
    Ok((out.len() == wo.len()).then_some(out))
}

/// `log₂ g!`: the bits of a pairing of `g` listed groups with `g` others.
fn pairing_bits(g: usize) -> f64 {
    (2..=g).map(|i| (i as f64).log2()).sum()
}

/// The rotation-scaling placements of matched two-column groups (`reflections` per pair), the
/// diagonal of the one-column ones.
fn rotation_placements(pairs: &[(usize, usize, usize)], reflections: &[bool]) -> Vec<Placement> {
    let mut out = Vec::new();
    let mut flags = reflections.iter();
    for &(w, r, width) in pairs {
        if width == 2 {
            let reflect = *flags.next().unwrap_or(&false);
            if reflect {
                out.push(vec![(w, r, 1.0), (w + 1, r + 1, -1.0)]);
                out.push(vec![(w, r + 1, 1.0), (w + 1, r, 1.0)]);
            } else {
                out.push(vec![(w, r, 1.0), (w + 1, r + 1, 1.0)]);
                out.push(vec![(w, r + 1, -1.0), (w + 1, r, 1.0)]);
            }
        } else {
            for c in 0..width {
                out.push(vec![(w + c, r + c, 1.0)]);
            }
        }
    }
    out
}

/// One real per matched column.
fn diagonal_placements(pairs: &[(usize, usize, usize)]) -> Vec<Placement> {
    pairs.iter().flat_map(|&(w, r, width)| (0..width).map(move |c| vec![(w + c, r + c, 1.0)])).collect()
}

/// A prepared chart's groups in decreasing whitened energy of the block they can carry,
/// `tr(G_g⁺ T_g G T_gᵀ)` with `T_g = X_gᵀ factor` (`F uᵀ` and `G = v C vᵀ` on the writer, `C vᵀ` and
/// `u F uᵀ` on the reader); nothing for a chart that takes every group.
fn ranked(chart: &Prepared, factor: &Array2<f64>, other: &Array2<f64>) -> Vec<(usize, f64)> {
    if !chart.chart.subsets {
        return Vec::new();
    }
    let t = mm(&chart.chart.basis.t(), factor);
    let mut out: Vec<(usize, f64)> = chart
        .chart
        .groups
        .iter()
        .zip(&chart.group_inverse)
        .enumerate()
        .map(|(g, (group, inverse))| {
            let tg = t.slice(s![group.start..group.start + group.width, ..]);
            (g, (inverse * &tg.dot(other).dot(&tg.t())).sum())
        })
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// What every choice of charts in one description shares.
struct Context<'a> {
    block: &'a Block<'a>,
    rank: usize,
    charts_bits: f64,
    core_bits: f64,
}

impl Context<'_> {
    /// The cheapest core on the writer groups `wg` of chart `i` and the reader groups `rg` of chart
    /// `j` (the identity at index 0): generic of every rank up to the block's and, when the groups
    /// pair, rotation-scaling and diagonal.
    fn evaluate(&self, (i, writer, wg): (usize, &Prepared, &[usize]), (j, reader, rg): (usize, &Prepared, &[usize])) -> Result<Option<Description>, String> {
        let block = self.block;
        let sides = block.sides(writer, wg, reader, rg)?;
        let structure = self.charts_bits + self.core_bits + writer.chart.subset_bits(wg.len())? + reader.chart.subset_bits(rg.len())?;
        let label = |chart: &Chart, groups: &[usize]| (chart.name.clone(), groups.iter().map(|g| chart.groups[*g].mode.clone()).collect::<Vec<_>>());
        let finish = |coded: Coded, core: Core| {
            let (u, v) = sides.decoded(&coded.a, &coded.b);
            Description {
                writer: label(&writer.chart, wg),
                reader: label(&reader.chart, rg),
                core,
                reals: coded.reals,
                structure_bits: structure + coded.structure_bits,
                real_bits: coded.real_bits,
                kl_bits: coded.kl,
                u,
                v,
                choice: Some(Choice { writer: i, writer_groups: wg.to_vec(), reader: j, reader_groups: rg.to_vec(), exponents: coded.exponents }),
            }
        };
        let mut best: Option<Description> = None;
        let mut offer = |candidate: Description| {
            if best.as_ref().is_none_or(|b| candidate.total() < b.total()) {
                best = Some(candidate);
            }
        };
        for coded in generic_cores(block, &sides, self.rank)? {
            let rank = coded.a.ncols();
            offer(finish(coded, Core::Generic { rank }));
        }
        if i == 0 || j == 0 {
            return Ok(best);
        }
        let Some(pairs) = paired(&writer.chart, wg, &reader.chart, rg, &sides)? else { return Ok(best) };
        let planes = pairs.iter().filter(|p| p.2 == 2).count();
        // The pairing, and a reflection flag per plane.
        let flags = pairing_bits(pairs.len()) + planes as f64;
        let mut reflections = vec![false; planes];
        let mut current = linear_core(block, &sides, &rotation_placements(&pairs, &reflections), flags, None)?;
        for plane in 0..planes {
            reflections[plane] = true;
            let trial = linear_core(block, &sides, &rotation_placements(&pairs, &reflections), flags, None)?;
            let better = match (&trial, &current) {
                (Some(t), Some(c)) => t.total() < c.total(),
                (Some(_), None) => true,
                _ => false,
            };
            if better {
                current = trial;
            } else {
                reflections[plane] = false;
            }
        }
        if let Some(coded) = current {
            offer(finish(coded, Core::Rotation { reflections }));
        }
        if let Some(coded) = linear_core(block, &sides, &diagonal_placements(&pairs), pairing_bits(pairs.len()), None)? {
            offer(finish(coded, Core::Diagonal));
        }
        Ok(best)
    }
}

/// A chart's candidate group sets: the prefixes of its ranked groups with fewer columns than the
/// side is wide (beyond that its columns are dependent and the identity says the same map in fewer
/// reals), each with its column count and the energy its groups carry (each group's own, so their
/// sum estimates the set's); a chart that takes every group has the one set, carrying `whole`.
fn prefixes(chart: &Chart, ranked: &[(usize, f64)], whole: f64) -> Vec<(Vec<usize>, usize, f64)> {
    if !chart.subsets {
        return vec![((0..chart.groups.len()).collect(), chart.basis.ncols(), whole)];
    }
    let d = chart.basis.nrows();
    let mut out = Vec::new();
    let (mut columns, mut energy) = (0, 0.0);
    for n in 1..=ranked.len() {
        columns += chart.groups[ranked[n - 1].0].width;
        energy += ranked[n - 1].1;
        if columns >= d {
            break;
        }
        out.push((ranked[..n].iter().map(|(g, _)| *g).collect(), columns, energy.min(whole)));
    }
    out
}

/// The minimum of `f` over `0..len` by ternary search on the prefix length (the total is unimodal
/// in it to the order the search relies on: description bits grow with every group added, the error
/// they leave falls with diminishing returns), each value computed once; `None` when every value is
/// infinite.
fn minimize(len: usize, f: &mut dyn FnMut(usize) -> Result<f64, String>) -> Result<Option<usize>, String> {
    let mut memo: Vec<Option<f64>> = vec![None; len];
    let mut at = |n: usize, f: &mut dyn FnMut(usize) -> Result<f64, String>| -> Result<f64, String> {
        if let Some(v) = memo[n] {
            return Ok(v);
        }
        let v = f(n)?;
        memo[n] = Some(v);
        Ok(v)
    };
    let (mut lo, mut hi) = (0, len);
    while hi - lo > 3 {
        let (a, b) = (lo + (hi - lo) / 3, hi - 1 - (hi - lo) / 3);
        if at(a, f)? <= at(b, f)? {
            hi = b;
        } else {
            lo = a + 1;
        }
    }
    let mut best: Option<(usize, f64)> = None;
    for n in lo..hi {
        let v = at(n, f)?;
        if v.is_finite() && best.is_none_or(|(_, b)| v < b) {
            best = Some((n, v));
        }
    }
    Ok(best.map(|(n, _)| n))
}

/// A chart's Gram `Xᵀ G X` in its side's metric, formed once per site: whole when the chart has no
/// more columns than the side is wide (a choice of groups slices it), else as `G X` (`d × q`, the
/// smaller of the two), a choice of `k` columns then costing `d k²`.
enum Gram {
    Whole(Array2<f64>),
    Weighted(Array2<f64>),
}

/// A chart prepared on its side's metric: its Gram, each group's pseudo-inverse Gram and, for the
/// identity, the metric's pseudo-inverse root.
struct Prepared {
    chart: Chart,
    identity: bool,
    gram: Gram,
    group_inverse: Vec<Array2<f64>>,
    root: Option<Array2<f64>>,
}

impl Prepared {
    fn new(chart: Chart, metric: &Array2<f64>, identity: bool) -> Result<Self, String> {
        if identity {
            let root = Some(inverses(metric)?.1);
            return Ok(Self { chart, identity, gram: Gram::Whole(Array2::zeros((0, 0))), group_inverse: Vec::new(), root });
        }
        let weighted = mm(metric, &chart.basis);
        let gram = if chart.basis.ncols() <= chart.basis.nrows() { Gram::Whole(symmetric(&mm(&chart.basis.t(), &weighted))) } else { Gram::Weighted(weighted) };
        let mut prepared = Self { chart, identity, gram, group_inverse: Vec::new(), root: None };
        prepared.group_inverse = (0..prepared.chart.groups.len())
            .map(|g| inverses(&prepared.gram(&prepared.chart.indices(&[g]))).map(|x| x.0))
            .collect::<Result<_, _>>()?;
        Ok(prepared)
    }

    /// The Gram of the basis columns `at`.
    fn gram(&self, at: &[usize]) -> Array2<f64> {
        match &self.gram {
            Gram::Whole(gram) => gram.select(Axis(0), at).select(Axis(1), at),
            Gram::Weighted(weighted) => symmetric(&mm(&self.chart.basis.select(Axis(1), at).t(), &weighted.select(Axis(1), at))),
        }
    }
}

/// A site prepared for describing its blocks: its metric and its charts on each side (the identity
/// first).
pub struct Geometry {
    pub metric: Metric,
    writers: Vec<Prepared>,
    readers: Vec<Prepared>,
}

impl Geometry {
    pub fn new(metric: Metric, writers: Vec<Chart>, readers: Vec<Chart>) -> Result<Self, String> {
        let (d_out, d_in) = (metric.fisher.nrows(), metric.moment.nrows());
        let mut w = vec![Prepared::new(Chart::identity(d_out), &metric.fisher, true)?];
        for chart in writers {
            w.push(Prepared::new(chart, &metric.fisher, false)?);
        }
        let mut r = vec![Prepared::new(Chart::identity(d_in), &metric.moment, true)?];
        for chart in readers {
            r.push(Prepared::new(chart, &metric.moment, false)?);
        }
        Ok(Self { metric, writers: w, readers: r })
    }

    /// The cheapest description of the block `uᵀ v` (`u` is `r × d_out`, `v` is `r × d_in`), of rank
    /// at most `r`, over the identity and the site's charts: least description bits plus the KL bits
    /// of its error.
    /// Per pair of charts, the two sides' group sets are chosen by coordinate descent over their
    /// ranked prefixes, a scan ending where the structure bits and one bit a paired column already
    /// exceed the best total.
    pub fn describe(&self, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Description, String> {
        self.describe_at(u, v, 1.0, true)
    }

    /// The description whose error is priced by the exact KL `exact` gives (bits, at the metric's
    /// `n`) of a decoded candidate, where the second-order price is only a local model: a confident
    /// network's Fisher vanishes, and an error past its quadratic regime costs far more than it
    /// predicts. Each round describes at the metric's price times a calibration, measures the
    /// decoded description exactly, and scales the calibration by measured over predicted; every
    /// round's description is exactly priced, and the cheapest is returned with its measured error.
    /// The rounds end when the prediction holds to within a factor of two (or after eight).
    pub fn describe_exact(
        &self,
        u: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
        exact: &mut dyn FnMut(&Description) -> Result<f64, String>,
    ) -> Result<Description, String> {
        // Every family, then the identity charts alone: a chart whose error the local metric cannot
        // see (a reader outside a ReLU's active set, a direction only the decoded upstream excites)
        // is never repaired by a larger price, while the identity converges as precision grows.
        let mut best: Option<Description> = None;
        // The identity-only pass repeats the first when the site has no other chart.
        let passes: &[bool] = if self.writers.len() == 1 && self.readers.len() == 1 { &[true] } else { &[true, false] };
        for &charts in passes {
            let mut calibration = 1.0_f64;
            let mut last = f64::NAN;
            for _ in 0..8 {
                let mut d = self.describe_at(u, v, calibration, charts)?;
                let predicted = d.kl_bits / calibration;
                let measured = exact(&d)?.max(0.0);
                d.kl_bits = measured;
                if best.as_ref().is_none_or(|b| d.total() < b.total()) {
                    best = Some(d);
                }
                // Within a factor of two of its prediction, or not moved by a larger price.
                if measured <= 2.0 * predicted + 1.0 || measured == last {
                    break;
                }
                last = measured;
                // A prediction of nothing says only that the price must grow: at most by 2^20 a
                // round, so the scale stays finite.
                calibration *= (measured / predicted.max(f64::MIN_POSITIVE)).min(1048576.0);
            }
        }
        best.ok_or_else(|| "no description".to_string())
    }

    /// The bits naming the two charts, and the core's kind (generic, rotation, diagonal).
    fn header_bits(&self) -> Result<(f64, f64), String> {
        let charts = fixed_index_len_bits(self.writers.len()).map_err(|e| e.to_string())? as f64
            + fixed_index_len_bits(self.readers.len()).map_err(|e| e.to_string())? as f64;
        Ok((charts, fixed_index_len_bits(3).map_err(|e| e.to_string())? as f64))
    }

    /// The block `uᵀ v` sent in `previous`'s family at its precision, with no search: `None` when
    /// that family no longer holds it (a rank it no longer resolves, a pairing or pivot that fails)
    /// or `previous` is the empty description.
    pub fn recode(&self, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>, previous: &Description, calibration: f64) -> Result<Option<Description>, String> {
        let Some(choice) = &previous.choice else { return Ok(None) };
        let block = Block::new(u, v, &self.metric, calibration);
        let (writer, reader) = (&self.writers[choice.writer], &self.readers[choice.reader]);
        let (wg, rg) = (choice.writer_groups.as_slice(), choice.reader_groups.as_slice());
        let sides = block.sides(writer, wg, reader, rg)?;
        let coded = match &previous.core {
            Core::Generic { rank } => generic_core_at(&block, &sides, *rank, &choice.exponents)?,
            core => {
                let Some(pairs) = paired(&writer.chart, wg, &reader.chart, rg, &sides)? else { return Ok(None) };
                let (placements, flags) = match core {
                    Core::Rotation { reflections } => {
                        let planes = pairs.iter().filter(|p| p.2 == 2).count();
                        (rotation_placements(&pairs, reflections), pairing_bits(pairs.len()) + planes as f64)
                    }
                    _ => (diagonal_placements(&pairs), pairing_bits(pairs.len())),
                };
                linear_core(&block, &sides, &placements, flags, choice.exponents.first().copied())?
            }
        };
        let Some(coded) = coded else { return Ok(None) };
        let (charts_bits, core_bits) = self.header_bits()?;
        let structure = charts_bits + core_bits + writer.chart.subset_bits(wg.len())? + reader.chart.subset_bits(rg.len())?;
        let (du, dv) = sides.decoded(&coded.a, &coded.b);
        Ok(Some(Description {
            writer: previous.writer.clone(),
            reader: previous.reader.clone(),
            core: previous.core.clone(),
            reals: coded.reals,
            structure_bits: structure + coded.structure_bits,
            real_bits: coded.real_bits,
            kl_bits: coded.kl,
            u: du,
            v: dv,
            choice: Some(Choice { exponents: coded.exponents, ..choice.clone() }),
        }))
    }

    /// [`Geometry::describe`] with the metric's price scaled by `calibration`.
    /// With `charts` false, only the identity charts (the generic family).
    fn describe_at(&self, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>, calibration: f64, charts: bool) -> Result<Description, String> {
        let metric = &self.metric;
        let block = Block::new(u, v, metric, calibration);
        let rank = u.nrows();
        let (charts_bits, core_bits) = self.header_bits()?;
        let writer_sets: Vec<_> = self.writers.iter().map(|c| prefixes(&c.chart, &ranked(c, &block.fu, &block.gv), block.w2)).collect();
        let reader_sets: Vec<_> = self.readers.iter().map(|c| prefixes(&c.chart, &ranked(c, &block.cv, &block.gu), block.w2)).collect();
        let context = Context { block: &block, rank, charts_bits, core_bits };
        let mut best: Option<Description> = None;
        let used = |n: usize| if charts { n } else { 1 };
        // The identity pair first, coded exactly; every other pair at the sets its proxy prefers (the
        // groups' bits, the identity's bits per real for every real, and the energy the sets leave out
        // at the metric's price), then coded exactly there.
        for (i, writer) in self.writers.iter().enumerate().take(used(self.writers.len())) {
            for (j, reader) in self.readers.iter().enumerate().take(used(self.readers.len())) {
                let per_real = best.as_ref().map_or(1.0, |b| b.real_bits / b.reals.max(1) as f64);
                let mut choice: Option<(f64, &Vec<usize>, &Vec<usize>)> = None;
                for (wg, ws, we) in &writer_sets[i] {
                    for (rg, rs, re) in &reader_sets[j] {
                        let r = rank.min(*ws).min(*rs);
                        let proxy = writer.chart.subset_bits(wg.len())?
                            + reader.chart.subset_bits(rg.len())?
                            + per_real * (r * (ws + rs - r)) as f64
                            + block.scale * ((block.w2 - we).max(0.0) + (block.w2 - re).max(0.0));
                        if choice.is_none_or(|c| proxy < c.0) {
                            choice = Some((proxy, wg, rg));
                        }
                    }
                }
                let Some((_, wg, rg)) = choice else { continue };
                if let Some(found) = context.evaluate((i, writer, wg.as_slice()), (j, reader, rg.as_slice()))?
                    && best.as_ref().is_none_or(|b| found.total() < b.total())
                {
                    best = Some(found);
                }
            }
        }
        // A map no family resolves (numerically zero in the metric) is the empty description:
        // its family's index alone, its whole map left as error.
        Ok(best.unwrap_or_else(|| Description {
            writer: ("identity".to_string(), Vec::new()),
            reader: ("identity".to_string(), Vec::new()),
            core: Core::Generic { rank: 0 },
            reals: 0,
            structure_bits: charts_bits + core_bits,
            real_bits: 0.0,
            kl_bits: block.w2.max(0.0) * block.scale,
            u: Array2::zeros((1, u.ncols())),
            v: Array2::zeros((1, v.ncols())),
            choice: None,
        }))
    }
}

/// The cheapest description of the block `w` (`d_out × d_in`, rank at most `rank`) over the identity
/// charts and `writers` / `readers`
/// ([`Geometry::describe`] on `w`'s leading `rank` singular pairs).
pub fn describe(w: &Array2<f64>, rank: usize, metric: &Metric, writers: &[Chart], readers: &[Chart]) -> Result<Description, String> {
    let geometry = Geometry::new(metric.clone(), writers.to_vec(), readers.to_vec())?;
    let decomposed = svd(w.view(), false).map_err(|e| format!("{e:?}"))?;
    let r = rank.min(decomposed.singular_values.iter().filter(|x| **x > decomposed.band).count()).max(1);
    let mut u = decomposed.u.slice(s![.., ..r]).t().to_owned();
    for (i, mut row) in u.rows_mut().into_iter().enumerate() {
        row *= decomposed.singular_values[i];
    }
    let v = decomposed.vt.slice(s![..r, ..]).to_owned();
    geometry.describe(u.view(), v.view())
}

/// `(M^{1/2}, M^{+1/2})` of a symmetric positive semidefinite matrix over its eigenvalues beyond the
/// band.
fn roots(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
    let d = eigh(symmetric(m).view(), SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let n = m.nrows();
    let (mut root, mut inverse) = (Array2::<f64>::zeros((n, n)), Array2::<f64>::zeros((n, n)));
    for (i, l) in d.values.iter().enumerate() {
        if *l > d.band {
            let q = d.vectors.column(i);
            let outer = q.insert_axis(Axis(1)).dot(&q.insert_axis(Axis(0)));
            root.scaled_add(l.sqrt(), &outer);
            inverse.scaled_add(1.0 / l.sqrt(), &outer);
        }
    }
    Ok((root, inverse))
}


/// Per site, the Gauss-Newton metric of its written value in logit space, `E[Jᵀ J]` with `J` the
/// Jacobian from the written value to the logits' shift-free part (`draws` Rademacher cotangents,
/// centred per row, pulled back by the exact reverse pass). Unlike the sampled-label Fisher, it does
/// not vanish where the network is confident: every direction that moves a logit is priced, and the
/// exact KL calibrates its scale ([`Geometry::describe_exact`]). In the shape of
/// [`super::pieces::Site::fisher`], so it can stand in for the sampled-label Fisher of
/// [`super::masked::site_statistics`].
pub fn logit_gauss_newton(
    program: &super::operator_program::OperatorProgram,
    chosen: &[super::masked::Site],
    family: &super::operator_program::FamilyInputs,
    trace: &super::operator_program::Trace,
    draws: usize,
) -> Result<Vec<Array2<f64>>, String> {
    let logits = &trace.values[program.output];
    let (rows, classes) = logits.dim();
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    let mut out: Vec<Array2<f64>> = chosen
        .iter()
        .map(|s| {
            let d: usize = s.writes.iter().map(|n| interfaces[*n].width()).sum();
            Array2::zeros((d, d))
        })
        .collect();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..draws {
        let mut cotangent = Array2::<f64>::zeros((rows, classes));
        for mut row in cotangent.rows_mut() {
            for x in row.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *x = if state & 1 == 0 { 1.0 } else { -1.0 };
            }
            let mean = row.sum() / classes as f64;
            row.mapv_inplace(|x| x - mean);
        }
        let back = super::derivatives::vjp(program, family, trace, cotangent).map_err(|e| e.to_string())?;
        for (site, metric) in chosen.iter().zip(out.iter_mut()) {
            let parts: Vec<Array2<f64>> =
                site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
            let views: Vec<_> = parts.iter().map(|x| x.view()).collect();
            let g = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
            *metric += &g.t().dot(&g);
        }
    }
    for metric in out.iter_mut() {
        *metric /= (draws * rows) as f64;
    }
    Ok(out)
}

/// [`Geometry::describe`] as the blocks' description ([`super::blocks::Describe`]): a block costs its
/// cheapest description plus the price of that description's error, the metric's price times
/// `calibration` (the ratio of measured to priced rounding KL, [`super::blocks::rounding_error`]).
pub struct Structured {
    pub sites: Vec<Geometry>,
    pub calibration: f64,
    /// Per `(site, index)`, the last description given ([`Structured::describe_cached`]).
    cache: Mutex<HashMap<(usize, usize), Description>>,
}

impl Structured {
    pub fn new(sites: Vec<Geometry>) -> Self {
        Self { sites, calibration: 1.0, cache: Mutex::new(HashMap::new()) }
    }

    /// The description of block `index` of site `site`, from its last one when it can: re-sent in
    /// that family at that precision ([`Geometry::recode`]), and searched afresh only when the
    /// family no longer holds the block or the re-sent total has grown by more than the family's own
    /// structure bits (all a fresh search could re-allocate without changing the reals' count). A
    /// library that trains by small steps is so re-priced at the cost of one coding, not a search.
    pub fn describe_cached(&self, site: usize, index: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Description, String> {
        let geometry = self.sites.get(site).ok_or_else(|| format!("no site {site}"))?;
        let previous = self.cache.lock().map_err(|e| e.to_string())?.get(&(site, index)).cloned();
        if let Some(previous) = previous
            && let Some(d) = geometry.recode(u, v, &previous, self.calibration)?
            && d.total() <= previous.total() + previous.structure_bits
        {
            self.cache.lock().map_err(|e| e.to_string())?.insert((site, index), d.clone());
            return Ok(d);
        }
        let d = self.describe(site, u, v)?;
        self.cache.lock().map_err(|e| e.to_string())?.insert((site, index), d.clone());
        Ok(d)
    }

    /// The same description with its error price scaled by `ratio`.
    pub fn scaled(mut self, ratio: f64) -> Self {
        self.calibration *= ratio;
        self
    }

    fn describe(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Description, String> {
        let geometry = self.sites.get(site).ok_or_else(|| format!("no site {site}"))?;
        geometry.describe_at(u, v, self.calibration, true)
    }
}

impl super::blocks::Describe for Structured {
    fn bits(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        if u.nrows() == 0 {
            return Ok(0.0);
        }
        Ok(self.describe(site, u, v)?.total())
    }

    fn bits_at(&self, site: usize, index: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        if u.nrows() == 0 {
            return Ok(0.0);
        }
        Ok(self.describe_cached(site, index, u, v)?.total())
    }

    fn decode(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Option<(Array2<f64>, Array2<f64>, f64)>, String> {
        let d = self.describe(site, u, v)?;
        Ok(Some((d.u, d.v, d.kl_bits)))
    }
}

/// [`Structured`] priced at the exact KL of its decoded blocks ([`Geometry::describe_exact`]): a
/// block's error is measured by the masked forward of the native model with only that block's map
/// replaced by its decoded description (every site otherwise its native map, every word on), per
/// word at `n`; on a site whose writes reach the logits linearly, by the certified bracket of
/// [`Exact::certified`] when it is tight enough, with no forward. Context-free, so the blocks' code can call it for any block it proposes; the price
/// of running only on some words is the per-word error over all of them.
pub struct Exact<'a> {
    pub structured: Structured,
    model: &'a super::operator_program::OperatorProgram,
    sites: Vec<super::masked::Site>,
    batches: Vec<(super::operator_program::FamilyInputs, super::masked::Target)>,
    observations: f64,
    maps: Vec<Array2<f64>>,
    base: f64,
    /// Per site, whether its written values reach the logits only through affine nodes, and per
    /// batch the native trace (the reads and the base of the linear change).
    linear: Vec<bool>,
    traces: Vec<super::operator_program::Trace>,
}

/// `(e^R − 1 − R)/R²` and `(e^{−R} − 1 + R)/R²` (both ½ at `R = 0`).
fn sandwich(r: f64) -> (f64, f64) {
    if r < 1e-4 {
        (0.5 + r / 6.0, 0.5 - r / 6.0)
    } else {
        ((r.exp_m1() - r) / (r * r), ((-r).exp_m1() + r) / (r * r))
    }
}

impl<'a> Exact<'a> {
    pub fn new(
        structured: Structured,
        model: &'a super::operator_program::OperatorProgram,
        sites: Vec<super::masked::Site>,
        batches: Vec<(super::operator_program::FamilyInputs, super::masked::Target)>,
        observations: f64,
    ) -> Result<Self, String> {
        let maps = sites.iter().map(|s| super::masked::matrix(model, s)).collect::<Result<Vec<_>, _>>()?;
        // A node is linear in a site's written values when every node between them and it is affine.
        let linear = sites
            .iter()
            .map(|site| {
                let mut reached = vec![false; model.nodes.len()];
                let mut affine = true;
                for (index, node) in model.nodes.iter().enumerate() {
                    reached[index] = site.writes.contains(&index) || node.arguments().iter().any(|a| reached[*a]);
                    if reached[index] && !site.writes.contains(&index) {
                        affine &= matches!(
                            node,
                            super::operator_program::Node::Affine { .. }
                                | super::operator_program::Node::Readout { .. }
                                | super::operator_program::Node::Concat { .. }
                        );
                    }
                }
                affine
            })
            .collect();
        let traces = batches.iter().map(|(inputs, _)| model.execute(inputs, false).map_err(|e| e.to_string())).collect::<Result<Vec<_>, _>>()?;
        let mut exact = Self { structured, model, sites, batches, observations, maps, base: 0.0, linear, traces };
        exact.base = exact.kl_bits(None)?;
        Ok(exact)
    }

    /// The KL bits per word of the native model with site `k`'s map changed by `− uᵀv + ũᵀṽ`.
    fn kl_bits(&self, change: Option<(usize, ArrayView2<'_, f64>, ArrayView2<'_, f64>, &Description)>) -> Result<f64, String> {
        let mut libraries = Vec::new();
        for (j, w) in self.maps.iter().enumerate() {
            let (d_out, d_in) = w.dim();
            let (mut u, mut v) = (Array2::<f64>::eye(d_out), w.clone());
            if let Some((_, bu, bv, d)) = change.as_ref().filter(|c| c.0 == j) {
                u = ndarray::concatenate(Axis(0), &[u.view(), (-&bu.to_owned()).view(), d.u.view()]).map_err(|e| e.to_string())?;
                v = ndarray::concatenate(Axis(0), &[v.view(), bv.view(), d.v.view()]).map_err(|e| e.to_string())?;
            }
            libraries.push(super::masked::Library { v, u, mean: Array1::zeros(d_in) });
        }
        let ranks: Vec<Vec<usize>> = libraries.iter().map(|l| vec![l.u.nrows()]).collect();
        let masked = super::masked::Masked::build_blocks(self.model, self.sites.clone(), libraries, ranks)?;
        let (mut total, mut rows) = (0.0_f64, 0.0_f64);
        for (inputs, target) in &self.batches {
            let masks: Vec<Array2<f64>> = self.sites.iter().map(|_| Array2::ones((inputs.rows, 1))).collect();
            let (kl, _, _) = super::masked::forward(&masked, &masked.family(inputs, &masks), target)?;
            for r in (0..inputs.rows).filter(|r| target.scores(*r)) {
                total += kl[r];
                rows += 1.0;
            }
        }
        Ok(self.observations * total / rows.max(1.0) / std::f64::consts::LN_2)
    }

    /// On a site whose written values reach the logits linearly, the KL of a change `δ` in the
    /// logits is bracketed without a forward (module note, "Pricing the error"): with `σ²` its
    /// variance under the native output and `R` its oscillation, `σ² c₋(R) ≤ KL ≤ log(1 + σ² c₊(R))`.
    /// The upper bound in bits, when the bracket is within the factor two
    /// [`Geometry::describe_exact`] accepts; `None` otherwise or off a linear site.
    pub fn certified(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>, d: &Description) -> Result<Option<f64>, String> {
        if !self.linear[site] {
            return Ok(None);
        }
        let change = d.u.t().dot(&d.v) - u.t().dot(&v);
        let interfaces = self.model.interfaces().map_err(|e| e.to_string())?;
        let (mut lower, mut upper, mut rows) = (0.0_f64, 0.0_f64, 0.0_f64);
        for ((inputs, target), trace) in self.batches.iter().zip(&self.traces) {
            let x = super::masked::read_values(trace, &self.sites[site])?;
            let written = x.dot(&change.t());
            let mut seeds = std::collections::BTreeMap::new();
            let mut at = 0;
            for n in &self.sites[site].writes {
                let w = interfaces[*n].width();
                seeds.insert(*n, written.slice(s![.., at..at + w]).to_owned());
                at += w;
            }
            let delta = super::derivatives::jvp_seeded(self.model, inputs, trace, &std::collections::BTreeMap::new(), &seeds).map_err(|e| e.to_string())?;
            for r in (0..inputs.rows).filter(|r| target.scores(*r)) {
                let z = target.logits.row(r);
                let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = z.iter().map(|v| (v - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                let dr = delta.row(r);
                let mean: f64 = weights.iter().zip(dr.iter()).map(|(w, x)| w * x).sum::<f64>() / total;
                let variance: f64 = weights.iter().zip(dr.iter()).map(|(w, x)| w * (x - mean) * (x - mean)).sum::<f64>() / total;
                let range = dr.iter().copied().fold(f64::NEG_INFINITY, f64::max) - dr.iter().copied().fold(f64::INFINITY, f64::min);
                let (plus, minus) = sandwich(range);
                lower += variance * minus;
                upper += (variance * plus).ln_1p();
                rows += 1.0;
            }
        }
        let scale = self.observations / rows.max(1.0) / std::f64::consts::LN_2;
        Ok((upper <= 2.0 * lower + f64::EPSILON).then_some(upper * scale))
    }

    /// The error bits per word of `d` replacing the block `uᵀv` of site `site`, by the forward.
    pub fn measured(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>, d: &Description) -> Result<f64, String> {
        Ok(self.kl_bits(Some((site, u, v, d)))? - self.base)
    }

    fn describe(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Description, String> {
        let geometry = self.structured.sites.get(site).ok_or_else(|| format!("no site {site}"))?;
        geometry.describe_exact(u, v, &mut |d| match self.certified(site, u, v, d)? {
            Some(bits) => Ok(bits),
            None => Ok(self.kl_bits(Some((site, u, v, d)))? - self.base),
        })
    }
}

impl super::blocks::Describe for Exact<'_> {
    fn bits(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        if u.nrows() == 0 {
            return Ok(0.0);
        }
        Ok(self.describe(site, u, v)?.total())
    }

    fn decode(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Option<(Array2<f64>, Array2<f64>, f64)>, String> {
        let d = self.describe(site, u, v)?;
        Ok(Some((d.u, d.v, d.kl_bits)))
    }
}

/// The charts a site's interfaces declare ([`Chart::coordinates`]): on each side whose nodes are
/// several (attention heads), one group per node; on a written node an attention reads as a query or
/// key under a rotary, one group per rotary plane. A block living on a few heads or planes names
/// them instead of sending every coordinate. `(writers, readers)`.
pub fn declared_charts(program: &super::operator_program::OperatorProgram, site: &super::masked::Site) -> Result<(Vec<Chart>, Vec<Chart>), String> {
    use super::operator_program::Node;
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    let nodes = |list: &[usize]| {
        let mut at = 0;
        list.iter()
            .map(|n| {
                let width = interfaces[*n].width();
                let group: Vec<usize> = (at..at + width).collect();
                at += width;
                group
            })
            .collect::<Vec<_>>()
    };
    let (written, read) = (nodes(&site.writes), nodes(&site.reads));
    let (d_out, d_in) = (written.iter().map(Vec::len).sum(), read.iter().map(Vec::len).sum());
    let mut writers = Vec::new();
    if written.len() > 1 {
        writers.push(Chart::coordinates("written nodes", d_out, &written)?);
    }
    let mut planes = Vec::new();
    for (n, group) in site.writes.iter().zip(&written) {
        let rotary = program.nodes.iter().find_map(|node| match node {
            Node::Attend { query, key, rotary: Some(r), .. } if query == n || key == n => Some(r.pairs()),
            _ => None,
        });
        if let Some(pairs) = rotary {
            planes.extend(pairs.into_iter().filter(|(a, b)| *a < group.len() && *b < group.len()).map(|(a, b)| vec![group[a], group[b]]));
        }
    }
    if !planes.is_empty() {
        writers.push(Chart::coordinates("rotary planes", d_out, &planes)?);
    }
    let mut readers = Vec::new();
    if read.len() > 1 {
        readers.push(Chart::coordinates("read nodes", d_in, &read)?);
    }
    Ok((writers, readers))
}

/// [`Structured`] for the exact price, with [`super::blocks::Generic`]'s closed form as the cheap
/// one a fit selects by where the exact price cannot change a set
/// ([`super::blocks::Describe::cheap`]).
pub struct Tiered {
    pub cheap: super::blocks::Generic,
    pub exact: Structured,
}

impl super::blocks::Describe for Tiered {
    fn bits(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        self.exact.bits(site, u, v)
    }

    fn cheap(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Option<f64>, String> {
        Ok(Some(self.cheap.bits(site, u, v)?))
    }

    fn bits_at(&self, site: usize, index: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<f64, String> {
        self.exact.bits_at(site, index, u, v)
    }

    fn decode(&self, site: usize, u: ArrayView2<'_, f64>, v: ArrayView2<'_, f64>) -> Result<Option<(Array2<f64>, Array2<f64>, f64)>, String> {
        self.exact.decode(site, u, v)
    }
}

// #2234 step 1a — the orbit-eliminated evidence on the arrow route, and the seams the exact-A
// log-determinant channels read their operands through, so the dense route and the arrow route
// run one channel implementation.
//
// Included into `construction.rs` beside `construction_orbit_elimination.rs`, whose dense
// stiffening this file prices without materializing the joint operator.
//
// # The bordered operator
//
// With `N = TᵀΦT` and the stiffened `A_s` of `construction_orbit_elimination.rs`, the bordered
//
//   K = [ A    ΦT ]
//       [ TᵀΦ  0  ]
//
// satisfies, in the basis `[T, Q]` with `QᵀΦT = 0`,
//
//   det A_s = (−1)^K·det K / det N,        In(K) = In(QᵀAQ) + (K, K, 0).
//
// `K` is an arrow: its rows are `A`'s own row blocks and its border carries `K` multipliers beside
// `β`. One arrow elimination therefore prices `log|A_s|` with no `A⁻¹`, which matters because `A`
// is singular along an exact symmetry and every `A⁻¹`-based identity cancels there. The top-left
// block of `K⁻¹` is the constrained inverse `G_⊥ = Q(QᵀAQ)⁻¹Qᵀ` and its off-diagonal block is
// `VN⁻¹`, `V = T − G_⊥AT`, so every term the dense route reads off its eliminated block is one
// solve with `K`.
//
// # The certificate
//
// `A_s − τΦ` is `diag((1 − τ)N, Qᵀ(A − τΦ)Q)` in `[T, Q]`, so for `τ < 1` the pencil directions of
// `(A_s, Φ)` below `τ` number `n₋(K_τ) − K`, `K_τ` the bordered `A − τΦ`: one elimination per
// shift. Step 1a prices exactly the states on which every direction clears the band edge the dense
// route classifies it by (`sae_exact_a_band_edge`); the rest refuse by name.
//
// # The seams
//
// Every penalty derivative `∂A/∂ρ` lives on the arrow's own positions: a row's coordinate block or
// the border block. The θ-adjoint reads its weight on those positions and on each row's
// coordinate–border block. A channel that writes its derivatives into a `PenaltyDerivativeSink`
// and reads its weight through a `JointWeight` is the same arithmetic whether the weight is held
// dense or as arrow blocks, and the arrow blocks of `A` are written by the dense materialization's
// own probes through an `ExactHessianProbeSink`, so both routes price one operator entry for
// entry.

#[cfg(test)]
#[path = "tests_orbit_arrow_2234.rs"]
mod tests_orbit_arrow_2234;

/// The exact-`A` geometry a bundle-free outer gradient reads its log-determinant channels off:
/// the dense route's spectral block, or the arrow orbit lane's bordered elimination.
#[derive(Clone, Copy)]
pub(crate) enum ExactAGeometry<'a> {
    Dense(&'a DenseExactAGeometry),
    ArrowOrbit(&'a ArrowOrbitGeometry),
}

impl<'a> ExactAGeometry<'a> {
    pub(crate) fn dense(self) -> Option<&'a DenseExactAGeometry> {
        match self {
            Self::Dense(geometry) => Some(geometry),
            Self::ArrowOrbit(_) => None,
        }
    }
}

/// A symmetric joint `(t, β)` weight read on the arrow's positions: within one row's coordinate
/// block, between a row's coordinates and the border, and within the border.
pub(crate) trait JointWeight {
    /// `W[row, column]` at global joint indices on one of those positions.
    fn entry(&self, row: usize, column: usize) -> f64;
    /// The coordinate block of the row starting at `base`, `q × q`.
    fn row_block(&self, base: usize, q: usize) -> Array2<f64>;
    /// The border block for a coordinate block of length `total_t`, or `None` when the weight was
    /// not laid out for that length.
    fn border_block(&self, total_t: usize) -> Option<ArrayView2<'_, f64>>;
    /// The dense matrix, when the weight is held dense. A channel reading cross-row entries (the
    /// ordered Beta--Bernoulli shared mass) needs it.
    fn dense(&self) -> Option<&Array2<f64>>;
}

impl<W: JointWeight + ?Sized> JointWeight for &W {
    fn entry(&self, row: usize, column: usize) -> f64 {
        (**self).entry(row, column)
    }

    fn row_block(&self, base: usize, q: usize) -> Array2<f64> {
        (**self).row_block(base, q)
    }

    fn border_block(&self, total_t: usize) -> Option<ArrayView2<'_, f64>> {
        (**self).border_block(total_t)
    }

    fn dense(&self) -> Option<&Array2<f64>> {
        (**self).dense()
    }
}

impl JointWeight for Array2<f64> {
    fn entry(&self, row: usize, column: usize) -> f64 {
        self[[row, column]]
    }

    fn row_block(&self, base: usize, q: usize) -> Array2<f64> {
        self.slice(s![base..base + q, base..base + q]).to_owned()
    }

    fn border_block(&self, total_t: usize) -> Option<ArrayView2<'_, f64>> {
        (total_t <= self.nrows()).then(|| self.slice(s![total_t.., total_t..]))
    }

    fn dense(&self) -> Option<&Array2<f64>> {
        Some(self)
    }
}

/// Where the penalty-derivative builders write `∂A/∂ρ_flat`, one entry or one row block at a time,
/// at global joint indices.
pub(crate) trait PenaltyDerivativeSink {
    fn add(&mut self, flat: usize, row: usize, column: usize, value: f64);
    fn add_row_block(&mut self, flat: usize, base: usize, block: &Array2<f64>);
    /// Record that `flat` carries a derivative operator, even one with no nonzero entry.
    fn touch(&mut self, flat: usize);
}

/// The dense `dim × dim` operator per flat coordinate the dense route contracts.
pub(crate) struct DensePenaltyDerivatives {
    dim: usize,
    pub(crate) by_flat: std::collections::BTreeMap<usize, Array2<f64>>,
}

impl DensePenaltyDerivatives {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            by_flat: std::collections::BTreeMap::new(),
        }
    }

    fn operator(&mut self, flat: usize) -> &mut Array2<f64> {
        let dim = self.dim;
        self.by_flat
            .entry(flat)
            .or_insert_with(|| Array2::<f64>::zeros((dim, dim)))
    }
}

impl PenaltyDerivativeSink for DensePenaltyDerivatives {
    fn add(&mut self, flat: usize, row: usize, column: usize, value: f64) {
        self.operator(flat)[[row, column]] += value;
    }

    fn add_row_block(&mut self, flat: usize, base: usize, block: &Array2<f64>) {
        let q = block.nrows();
        let mut target = self
            .operator(flat)
            .slice_mut(s![base..base + q, base..base + q]);
        target += block;
    }

    fn touch(&mut self, flat: usize) {
        self.operator(flat);
    }
}

/// `⟨W, ∂A/∂ρ_flat⟩` per flat coordinate, accumulated as the builders write, so no
/// per-coordinate operator is ever held.
pub(crate) struct ContractingPenaltyDerivatives<'w, W: JointWeight + ?Sized> {
    weight: &'w W,
    pub(crate) contractions: std::collections::BTreeMap<usize, f64>,
}

impl<'w, W: JointWeight + ?Sized> ContractingPenaltyDerivatives<'w, W> {
    pub(crate) fn new(weight: &'w W) -> Self {
        Self {
            weight,
            contractions: std::collections::BTreeMap::new(),
        }
    }
}

impl<W: JointWeight + ?Sized> PenaltyDerivativeSink for ContractingPenaltyDerivatives<'_, W> {
    fn add(&mut self, flat: usize, row: usize, column: usize, value: f64) {
        *self.contractions.entry(flat).or_insert(0.0) += self.weight.entry(row, column) * value;
    }

    fn add_row_block(&mut self, flat: usize, base: usize, block: &Array2<f64>) {
        let weight = self.weight.row_block(base, block.nrows());
        *self.contractions.entry(flat).or_insert(0.0) += (&weight * block).sum();
    }

    fn touch(&mut self, flat: usize) {
        self.contractions.entry(flat).or_insert(0.0);
    }
}

/// Where the exact-Hessian probes write the operator's columns: slot `c` of every row's coordinate
/// block from one probe, and each border column from its own (`materialize_exact_hessian_dense_with_gap_border`).
pub(crate) trait ExactHessianProbeSink {
    /// Column `slot` of the coordinate block `start..end`: `A[i, start + slot] = t[i]`.
    fn row_slot_column(&mut self, start: usize, end: usize, slot: usize, t: &Array1<f64>);
    /// The ordered Beta--Bernoulli mass Hessian `Σ c·uuᵀ`, whose carriers span rows.
    fn add_mass_carriers(&mut self, carriers: &[(f64, Vec<(usize, f64)>)]) -> Result<(), String>;
    /// Border column `j`: `A[i, total_t + j] = A[total_t + j, i] = column.t[i]` and
    /// `A[total_t + i, total_t + j] = column.beta[i]`.
    fn border_column(&mut self, total_t: usize, j: usize, column: &SaeArrowVector);
    /// Average every held entry with its transpose.
    fn symmetrize(&mut self);
}

impl ExactHessianProbeSink for Array2<f64> {
    fn row_slot_column(&mut self, start: usize, end: usize, slot: usize, t: &Array1<f64>) {
        let col = start + slot;
        for i in start..end {
            self[[i, col]] = t[i];
        }
    }

    fn add_mass_carriers(&mut self, carriers: &[(f64, Vec<(usize, f64)>)]) -> Result<(), String> {
        for (coefficient, carrier) in carriers {
            for &(row, left) in carrier {
                for &(col, right) in carrier {
                    self[[row, col]] += coefficient * left * right;
                }
            }
        }
        Ok(())
    }

    fn border_column(&mut self, total_t: usize, j: usize, column: &SaeArrowVector) {
        let col = total_t + j;
        for i in 0..total_t {
            self[[i, col]] = column.t[i];
            self[[col, i]] = column.t[i];
        }
        for i in 0..column.beta.len() {
            self[[total_t + i, col]] = column.beta[i];
        }
    }

    fn symmetrize(&mut self) {
        let dim = self.nrows();
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (self[[r, c]] + self[[c, r]]);
                self[[r, c]] = avg;
                self[[c, r]] = avg;
            }
        }
    }
}

/// A symmetric joint `(t, β)` operator held on the arrow's positions only: each row's coordinate
/// block, the coordinate–border block and the border block. Written by the exact-Hessian probes, it
/// holds the entries of the dense materialization bit for bit.
#[derive(Clone, Debug)]
pub(crate) struct ArrowJointBlocks {
    total_t: usize,
    k: usize,
    /// `n + 1` offsets: row `i` holds coordinate slots `row_offsets[i]..row_offsets[i + 1]`.
    row_offsets: Vec<usize>,
    /// The row owning each coordinate slot.
    slot_rows: Vec<usize>,
    rows: Vec<Array2<f64>>,
    /// The coordinate–border block, `total_t × k`.
    cross: Array2<f64>,
    border: Array2<f64>,
}

impl ArrowJointBlocks {
    pub(crate) fn zeros(row_offsets: &[usize], k: usize) -> Self {
        let n = row_offsets.len().saturating_sub(1);
        let total_t = row_offsets.last().copied().unwrap_or(0);
        let mut slot_rows = vec![0usize; total_t];
        let mut rows = Vec::with_capacity(n);
        for row in 0..n {
            let (start, end) = (row_offsets[row], row_offsets[row + 1]);
            slot_rows[start..end].fill(row);
            rows.push(Array2::<f64>::zeros((end - start, end - start)));
        }
        Self {
            total_t,
            k,
            row_offsets: row_offsets.to_vec(),
            slot_rows,
            rows,
            cross: Array2::<f64>::zeros((total_t, k)),
            border: Array2::<f64>::zeros((k, k)),
        }
    }

    fn dim(&self) -> usize {
        self.total_t + self.k
    }

    fn row_range(&self, row: usize) -> (usize, usize) {
        (self.row_offsets[row], self.row_offsets[row + 1])
    }

    /// `Φ`'s arrow blocks, read off `slots + k` applies of the metric exactly as the exact-Hessian
    /// probes read `A`'s: `Φ` couples no two rows, so one probe of slot `c` across every row returns
    /// column `c` of every row block.
    fn from_metric_probes(
        row_offsets: &[usize],
        k: usize,
        metric: &dyn ExactAPencilMetric,
    ) -> Result<Self, String> {
        let mut blocks = Self::zeros(row_offsets, k);
        let dim = blocks.dim();
        let total_t = blocks.total_t;
        let n = blocks.rows.len();
        let slots = (0..n).map(|row| blocks.rows[row].nrows()).max().unwrap_or(0);
        for slot in 0..slots {
            let mut unit = Array1::<f64>::zeros(dim);
            for row in 0..n {
                let (start, end) = blocks.row_range(row);
                if start + slot < end {
                    unit[start + slot] = 1.0;
                }
            }
            let image = metric.apply(unit.view())?;
            for row in 0..n {
                let (start, end) = blocks.row_range(row);
                if start + slot < end {
                    for i in start..end {
                        blocks.rows[row][[i - start, slot]] = image[i];
                    }
                }
            }
        }
        for j in 0..k {
            let mut unit = Array1::<f64>::zeros(dim);
            unit[total_t + j] = 1.0;
            let image = metric.apply(unit.view())?;
            for i in 0..total_t {
                blocks.cross[[i, j]] = image[i];
            }
            for i in 0..k {
                blocks.border[[i, j]] = image[total_t + i];
            }
        }
        blocks.symmetrize();
        Ok(blocks)
    }

    /// `‖·‖_F` of the symmetric operator, the coordinate–border block counted on both sides.
    fn frobenius(&self) -> f64 {
        let squares = |block: &Array2<f64>| block.iter().map(|value| value * value).sum::<f64>();
        let rows: f64 = self.rows.iter().map(squares).sum();
        (rows + 2.0 * squares(&self.cross) + squares(&self.border)).sqrt()
    }

    /// `self −= other`, block by block; both must hold the same layout.
    pub(crate) fn subtract(&mut self, other: &Self) -> Result<(), String> {
        if self.row_offsets != other.row_offsets || self.k != other.k {
            return Err("ArrowJointBlocks::subtract: the operators hold different layouts".to_string());
        }
        for (mine, theirs) in self.rows.iter_mut().zip(other.rows.iter()) {
            *mine -= theirs;
        }
        self.cross -= &other.cross;
        self.border -= &other.border;
        Ok(())
    }

    /// `self += other`, block by block; both must hold the same layout.
    pub(crate) fn accumulate(&mut self, other: &Self) -> Result<(), String> {
        if self.row_offsets != other.row_offsets || self.k != other.k {
            return Err(
                "ArrowJointBlocks::accumulate: the operators hold different layouts".to_string(),
            );
        }
        for (mine, theirs) in self.rows.iter_mut().zip(other.rows.iter()) {
            *mine += theirs;
        }
        self.cross += &other.cross;
        self.border += &other.border;
        Ok(())
    }

    /// `+= block` on row `row`'s coordinate block.
    pub(crate) fn add_to_row_block(&mut self, row: usize, block: &Array2<f64>) -> Result<(), String> {
        let target = self.rows.get_mut(row).ok_or_else(|| {
            format!("ArrowJointBlocks::add_to_row_block: no row {row}")
        })?;
        if target.dim() != block.dim() {
            return Err(format!(
                "ArrowJointBlocks::add_to_row_block: row {row} holds {:?}, the addend is {:?}",
                target.dim(),
                block.dim()
            ));
        }
        *target += block;
        Ok(())
    }

    /// `dense += self` on the joint `(t, β)` layout, the coordinate–border block on both sides.
    pub(crate) fn add_to_dense(&self, dense: &mut Array2<f64>) -> Result<(), String> {
        let (total_t, dim) = (self.total_t, self.dim());
        if dense.dim() != (dim, dim) {
            return Err(format!(
                "ArrowJointBlocks::add_to_dense: the operator has dimension {dim}, the dense \
                 matrix is {:?}",
                dense.dim()
            ));
        }
        for (row, block) in self.rows.iter().enumerate() {
            let (start, end) = self.row_range(row);
            let mut slice = dense.slice_mut(s![start..end, start..end]);
            slice += block;
        }
        {
            let mut upper = dense.slice_mut(s![..total_t, total_t..]);
            upper += &self.cross;
        }
        {
            let mut lower = dense.slice_mut(s![total_t.., ..total_t]);
            lower += &self.cross.t();
        }
        let mut border = dense.slice_mut(s![total_t.., total_t..]);
        border += &self.border;
        Ok(())
    }

    /// `⟨W, self⟩` over the arrow's positions, the coordinate–border block counted on both
    /// sides; every position off the arrow is zero in `self`.
    pub(crate) fn contract<W: JointWeight + ?Sized>(&self, weight: &W) -> Result<f64, String> {
        let total_t = self.total_t;
        let mut total = 0.0_f64;
        for (row, block) in self.rows.iter().enumerate() {
            let (start, end) = self.row_range(row);
            total += (&weight.row_block(start, end - start) * block).sum();
        }
        for i in 0..total_t {
            for c in 0..self.k {
                total += 2.0 * weight.entry(i, total_t + c) * self.cross[[i, c]];
            }
        }
        let border = weight.border_block(total_t).ok_or_else(|| {
            "ArrowJointBlocks::contract: the weight holds no border at this layout".to_string()
        })?;
        total += (&border * &self.border).sum();
        Ok(total)
    }

    /// The operator applied to a joint vector.
    fn apply(&self, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let total_t = self.total_t;
        let (x_t, x_beta) = (x.slice(s![..total_t]), x.slice(s![total_t..]));
        let mut out = Array1::<f64>::zeros(self.dim());
        for (row, block) in self.rows.iter().enumerate() {
            let (start, end) = self.row_range(row);
            out.slice_mut(s![start..end])
                .assign(&block.dot(&x_t.slice(s![start..end])));
        }
        {
            let mut coordinates = out.slice_mut(s![..total_t]);
            coordinates += &self.cross.dot(&x_beta);
        }
        let border = self.cross.t().dot(&x_t) + self.border.dot(&x_beta);
        out.slice_mut(s![total_t..]).assign(&border);
        out
    }

    /// `+= scale·sym(xyᵀ)` on the arrow's positions.
    fn add_symmetric_outer(&mut self, scale: f64, x: ArrayView1<'_, f64>, y: ArrayView1<'_, f64>) {
        let total_t = self.total_t;
        let half = 0.5 * scale;
        for row in 0..self.rows.len() {
            let (start, end) = self.row_range(row);
            let block = &mut self.rows[row];
            for a in start..end {
                for b in start..end {
                    block[[a - start, b - start]] += half * (x[a] * y[b] + y[a] * x[b]);
                }
            }
        }
        for i in 0..total_t {
            for c in 0..self.k {
                self.cross[[i, c]] += half * (x[i] * y[total_t + c] + y[i] * x[total_t + c]);
            }
        }
        for c in 0..self.k {
            for d in 0..self.k {
                self.border[[c, d]] +=
                    half * (x[total_t + c] * y[total_t + d] + y[total_t + c] * x[total_t + d]);
            }
        }
    }
}

impl JointWeight for ArrowJointBlocks {
    /// A cross-row coordinate entry is not held, and reads `NaN` so that no consumer can contract
    /// one silently.
    fn entry(&self, row: usize, column: usize) -> f64 {
        let total_t = self.total_t;
        match (row < total_t, column < total_t) {
            (true, true) => {
                let owner = self.slot_rows[row];
                if self.slot_rows[column] != owner {
                    return f64::NAN;
                }
                let base = self.row_offsets[owner];
                self.rows[owner][[row - base, column - base]]
            }
            (true, false) => self.cross[[row, column - total_t]],
            (false, true) => self.cross[[column, row - total_t]],
            (false, false) => self.border[[row - total_t, column - total_t]],
        }
    }

    fn row_block(&self, base: usize, q: usize) -> Array2<f64> {
        if q == 0 {
            return Array2::<f64>::zeros((0, 0));
        }
        let owner = self.slot_rows[base];
        if self.row_offsets[owner] != base || self.rows[owner].nrows() != q {
            return Array2::<f64>::from_elem((q, q), f64::NAN);
        }
        self.rows[owner].clone()
    }

    fn border_block(&self, total_t: usize) -> Option<ArrayView2<'_, f64>> {
        (total_t == self.total_t).then(|| self.border.view())
    }

    fn dense(&self) -> Option<&Array2<f64>> {
        None
    }
}

impl ExactHessianProbeSink for ArrowJointBlocks {
    fn row_slot_column(&mut self, start: usize, end: usize, slot: usize, t: &Array1<f64>) {
        let block = &mut self.rows[self.slot_rows[start]];
        for i in start..end {
            block[[i - start, slot]] = t[i];
        }
    }

    fn add_mass_carriers(&mut self, carriers: &[(f64, Vec<(usize, f64)>)]) -> Result<(), String> {
        if carriers.iter().any(|(_, carrier)| !carrier.is_empty()) {
            return Err(
                "arrow exact-Hessian blocks: the ordered Beta--Bernoulli mass Hessian couples rows, \
                 which no arrow block holds"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn border_column(&mut self, total_t: usize, j: usize, column: &SaeArrowVector) {
        for i in 0..total_t {
            self.cross[[i, j]] = column.t[i];
        }
        for i in 0..column.beta.len() {
            self.border[[i, j]] = column.beta[i];
        }
    }

    fn symmetrize(&mut self) {
        let average = |block: &mut Array2<f64>| {
            let dim = block.nrows();
            for r in 0..dim {
                for c in (r + 1)..dim {
                    let avg = 0.5 * (block[[r, c]] + block[[c, r]]);
                    block[[r, c]] = avg;
                    block[[c, r]] = avg;
                }
            }
        };
        for block in &mut self.rows {
            average(block);
        }
        average(&mut self.border);
    }
}

fn frobenius_of<S: ndarray::Data<Elem = f64>>(matrix: &ndarray::ArrayBase<S, ndarray::Ix2>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn symmetrized_in_place(matrix: &mut Array2<f64>) {
    let dim = matrix.nrows();
    for r in 0..dim {
        for c in (r + 1)..dim {
            let avg = 0.5 * (matrix[[r, c]] + matrix[[c, r]]);
            matrix[[r, c]] = avg;
            matrix[[c, r]] = avg;
        }
    }
}

/// One elimination of the bordered arrow `[[A − τΦ, C], [Cᵀ, 0]]`, `C` the `dim × m` extra border
/// columns: every row block by its own eigensystem, then the dense `(k + m)`-square Schur
/// complement by its.
struct BorderedArrowFactor {
    /// Each row block's eigensystem, held when the factor is kept for solves.
    rows: Vec<(Array1<f64>, Array2<f64>)>,
    /// `F = D⁻¹[A_tβ − τΦ_tβ, C_t]` stacked over rows, held when kept.
    coupled: Array2<f64>,
    schur_values: Array1<f64>,
    schur_vectors: Array2<f64>,
    log_abs_det: f64,
    negative: usize,
    /// The least resolved pivot as `|pivot| / its rounding scale`, and where it sits: below `1`
    /// the elimination cannot tell that pivot's sign.
    weakest_pivot: (f64, String),
}

impl BorderedArrowFactor {
    fn eliminate(
        operator: &ArrowJointBlocks,
        shift: Option<(&ArrowJointBlocks, f64)>,
        columns: ArrayView2<'_, f64>,
        keep: bool,
    ) -> Result<Self, String> {
        let (total_t, k) = (operator.total_t, operator.k);
        if columns.nrows() != operator.dim() {
            return Err(format!(
                "bordered arrow elimination: {} border columns of length {} on joint dimension {}",
                columns.ncols(),
                columns.nrows(),
                operator.dim()
            ));
        }
        let width = k + columns.ncols();
        let epsilon = f64::EPSILON;
        let mut schur = Array2::<f64>::zeros((width, width));
        schur.slice_mut(s![..k, ..k]).assign(&operator.border);
        if let Some((metric, tau)) = shift {
            schur.slice_mut(s![..k, ..k]).scaled_add(-tau, &metric.border);
        }
        schur
            .slice_mut(s![..k, k..])
            .assign(&columns.slice(s![total_t.., ..]));
        schur
            .slice_mut(s![k.., ..k])
            .assign(&columns.slice(s![total_t.., ..]).t());
        let mut schur_scale = frobenius_of(&schur);
        let mut rows = Vec::with_capacity(if keep { operator.rows.len() } else { 0 });
        let mut coupled = Array2::<f64>::zeros((if keep { total_t } else { 0 }, width));
        let mut log_abs_det = 0.0_f64;
        let mut negative = 0usize;
        let mut weakest_pivot = (f64::INFINITY, String::from("none"));
        let mut widest_row = 0usize;
        for (row, operator_block) in operator.rows.iter().enumerate() {
            let (start, end) = operator.row_range(row);
            let q = end - start;
            widest_row = widest_row.max(q);
            if q == 0 {
                if keep {
                    rows.push((Array1::<f64>::zeros(0), Array2::<f64>::zeros((0, 0))));
                }
                continue;
            }
            let mut block = operator_block.clone();
            if let Some((metric, tau)) = shift {
                block.scaled_add(-tau, &metric.rows[row]);
            }
            let (values, vectors) = block.eigh(Side::Lower).map_err(|error| {
                format!("bordered arrow elimination: row {row} eigendecomposition failed: {error:?}")
            })?;
            let scale = values.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
            let rounding = 8.0 * q as f64 * epsilon * scale;
            for &value in values.iter() {
                let resolved = if value == 0.0 {
                    0.0
                } else if rounding > 0.0 {
                    value.abs() / rounding
                } else {
                    f64::INFINITY
                };
                if resolved < weakest_pivot.0 {
                    weakest_pivot = (resolved, format!("row {row}'s coordinate block"));
                }
                log_abs_det += value.abs().ln();
                if value < 0.0 {
                    negative += 1;
                }
            }
            let mut coupling = Array2::<f64>::zeros((q, width));
            coupling
                .slice_mut(s![.., ..k])
                .assign(&operator.cross.slice(s![start..end, ..]));
            if let Some((metric, tau)) = shift {
                coupling
                    .slice_mut(s![.., ..k])
                    .scaled_add(-tau, &metric.cross.slice(s![start..end, ..]));
            }
            coupling
                .slice_mut(s![.., k..])
                .assign(&columns.slice(s![start..end, ..]));
            let mut projected = vectors.t().dot(&coupling);
            for (index, mut line) in projected.rows_mut().into_iter().enumerate() {
                line /= values[index];
            }
            let solved = vectors.dot(&projected);
            let update = coupling.t().dot(&solved);
            schur_scale += frobenius_of(&update);
            schur -= &update;
            if keep {
                coupled.slice_mut(s![start..end, ..]).assign(&solved);
                rows.push((values, vectors));
            }
        }
        symmetrized_in_place(&mut schur);
        let (schur_values, schur_vectors) = schur.eigh(Side::Lower).map_err(|error| {
            format!("bordered arrow elimination: reduced Schur eigendecomposition failed: {error:?}")
        })?;
        let rounding = 8.0 * (width + widest_row) as f64 * epsilon * schur_scale;
        for &value in schur_values.iter() {
            let resolved = if value == 0.0 {
                0.0
            } else if rounding > 0.0 {
                value.abs() / rounding
            } else {
                f64::INFINITY
            };
            if resolved < weakest_pivot.0 {
                weakest_pivot = (resolved, "the reduced Schur complement".to_string());
            }
            log_abs_det += value.abs().ln();
            if value < 0.0 {
                negative += 1;
            }
        }
        Ok(Self {
            rows,
            coupled,
            schur_values,
            schur_vectors,
            log_abs_det,
            negative,
            weakest_pivot,
        })
    }

    /// `K⁻¹[x_t; x_border]` on a kept factor: the coordinate part and the `k + m` border part.
    fn solve(
        &self,
        row_offsets: &[usize],
        x_t: ArrayView1<'_, f64>,
        x_border: ArrayView1<'_, f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut local = Array1::<f64>::zeros(x_t.len());
        for (row, (values, vectors)) in self.rows.iter().enumerate() {
            let (start, end) = (row_offsets[row], row_offsets[row + 1]);
            if start == end {
                continue;
            }
            let coefficients = vectors.t().dot(&x_t.slice(s![start..end])) / values;
            local
                .slice_mut(s![start..end])
                .assign(&vectors.dot(&coefficients));
        }
        let reduced = &x_border - &self.coupled.t().dot(&x_t);
        let coefficients = self.schur_vectors.t().dot(&reduced) / &self.schur_values;
        let border = self.schur_vectors.dot(&coefficients);
        let coordinates = local - self.coupled.dot(&border);
        (coordinates, border)
    }

    /// `S⁻¹` of the reduced Schur complement.
    fn schur_inverse(&self) -> Array2<f64> {
        let scaled = Array2::from_shape_fn(self.schur_vectors.raw_dim(), |(row, column)| {
            self.schur_vectors[[row, column]] / self.schur_values[column]
        });
        scaled.dot(&self.schur_vectors.t())
    }
}

/// A lower bound on `λ_min(Φ)` from its arrow blocks: with `Φ = [[T, C], [Cᵀ, B]]` and
/// `S = B − CᵀT⁻¹C`, `Φ⁻¹ = diag(T⁻¹, 0) + E S⁻¹ Eᵀ` for `E = [−T⁻¹C; I]`, so
/// `‖Φ⁻¹‖₂ ≤ ‖T⁻¹‖₂ + (1 + ‖T⁻¹C‖_F²)·‖S⁻¹‖₂`.
fn metric_smallest_eigenvalue_bound(metric: &ArrowJointBlocks) -> Result<f64, String> {
    let k = metric.k;
    let mut schur = metric.border.clone();
    let mut inverse_rows = 0.0_f64;
    let mut coupling_squares = 0.0_f64;
    for (row, block) in metric.rows.iter().enumerate() {
        let (start, end) = metric.row_range(row);
        if start == end {
            continue;
        }
        let (values, vectors) = block.eigh(Side::Lower).map_err(|error| {
            format!("arrow orbit lane: the metric's row {row} eigendecomposition failed: {error:?}")
        })?;
        let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
        if !(smallest.is_finite() && smallest > 0.0) {
            return Err(format!(
                "arrow orbit lane: the metric's row {row} block is not positive definite \
                 (smallest eigenvalue {smallest:e})"
            ));
        }
        inverse_rows = inverse_rows.max(1.0 / smallest);
        if k > 0 {
            let coupling = metric.cross.slice(s![start..end, ..]);
            let mut projected = vectors.t().dot(&coupling);
            for (index, mut line) in projected.rows_mut().into_iter().enumerate() {
                line /= values[index];
            }
            let solved = vectors.dot(&projected);
            coupling_squares += solved.iter().map(|value| value * value).sum::<f64>();
            schur -= &coupling.t().dot(&solved);
        }
    }
    let border_inverse = if k == 0 {
        0.0
    } else {
        symmetrized_in_place(&mut schur);
        let (values, _) = schur.eigh(Side::Lower).map_err(|error| {
            format!("arrow orbit lane: the metric's reduced Schur eigendecomposition failed: {error:?}")
        })?;
        let smallest = values.iter().copied().fold(f64::INFINITY, f64::min);
        if !(smallest.is_finite() && smallest > 0.0) {
            return Err(format!(
                "arrow orbit lane: the metric's reduced Schur is not positive definite \
                 (smallest eigenvalue {smallest:e})"
            ));
        }
        1.0 / smallest
    };
    Ok(1.0 / (inverse_rows + (1.0 + coupling_squares) * border_inverse))
}


/// The band edge a certified state's every pencil direction clears (#2234 step 1a).
///
/// The dense route retains a direction `w` of `(A_s, Φ)`, `wᵀΦw = 1`, when
/// `μ > max(√ε, dim·ε·‖w‖²(‖A_s‖_F + μ‖Φ‖_F))` on an unpinned factor (no substituted stiffness).
/// With `‖w‖² ≤ 1/λ_min(Φ)` and `c = dim·ε/λ_min(Φ)`, `μ > τ_cert = max(√ε, c‖A_s‖_F/(1 − c‖Φ‖_F))`
/// suffices. `margin` is the elimination's own rounding in `μ` units,
/// `γ_dim(‖A_s‖_F + τ_cert‖Φ‖_F + 2‖ΦT‖_F)/λ_min(Φ)`: the computed inertia is exact for a pencil
/// that close.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrowOrbitCertificate {
    /// `τ_cert`.
    pub threshold: f64,
    /// `δ`.
    pub margin: f64,
    /// The lower bound on `λ_min(Φ)` the threshold was derived with.
    pub metric_floor: f64,
    /// The upper bound on `‖A_s‖_F` the threshold was derived with.
    pub stiffened_frobenius: f64,
}

impl std::fmt::Display for ArrowOrbitCertificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "τ_cert = {:.6e}, margin δ = {:.3e}, λ_min(Φ) ≥ {:.3e}, ‖A_s‖_F ≤ {:.3e}",
            self.threshold, self.margin, self.metric_floor, self.stiffened_frobenius
        )
    }
}

/// #2234 step 1a — why the arrow route's orbit lane did not price a closure-certified circle
/// orbit. Every variant but [`Self::SurrogateLane`] is a refusal of the exact reduced-Schur lane.
#[derive(Clone, Debug, PartialEq)]
pub enum ArrowOrbitRefusal {
    /// The ordered Beta--Bernoulli prior couples rows through its shared mass, which no arrow block
    /// holds.
    CrossRowMass,
    /// Step 1a prices an unpinned evidence factor, `Φ = B_raw`.
    PinnedEvidenceFactor {
        gauge_directions: usize,
        row_pins: usize,
        conditioned_rows: usize,
        border_conditioning: bool,
        border_quotient: bool,
    },
    /// The exact lane's arrow blocks and its `width`-square reduced Schur exceed the in-core
    /// budget, and the rational surrogate that prices such a border carries no orbit pricing
    /// (#2234 step 2).
    SurrogateLane {
        bytes: usize,
        budget: usize,
        width: usize,
    },
    /// Negative inertia of the stiffened pencil: `count` resolved negative directions, which the
    /// dense route prices as a clamp basin or refuses as a saddle. The arrow route cannot tell the
    /// two apart, so it prices neither.
    NegativeInertia {
        count: usize,
        certificate: ArrowOrbitCertificate,
    },
    /// `count` directions in the band below the certificate's edge: a band-bearing orbit state,
    /// priced on the dense route only (#2234 step 1b).
    Band {
        count: usize,
        certificate: ArrowOrbitCertificate,
    },
    /// `count` directions within the certificate's own rounding margin of its edge, or, with
    /// `pivot`, an elimination pivot too small against its own rounding to sign.
    Undecided {
        count: usize,
        pivot: Option<(f64, String)>,
        certificate: ArrowOrbitCertificate,
    },
    /// The working coordinates resolve no band edge below the orbit's own unit curvature.
    Unresolvable { reason: String },
}

impl ArrowOrbitRefusal {
    /// The arrow lane that refused.
    pub fn lane(&self) -> &'static str {
        match self {
            Self::SurrogateLane { .. } => "surrogate lane",
            _ => "exact reduced-Schur orbit lane",
        }
    }
}

impl std::fmt::Display for ArrowOrbitRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CrossRowMass => f.write_str(
                "the ordered Beta--Bernoulli prior couples rows through its shared mass, which no \
                 arrow block holds",
            ),
            Self::PinnedEvidenceFactor {
                gauge_directions,
                row_pins,
                conditioned_rows,
                border_conditioning,
                border_quotient,
            } => write!(
                f,
                "step 1a prices an unpinned evidence factor, and this one carries \
                 {gauge_directions} gauge and {row_pins} row-direction pins, {conditioned_rows} \
                 spectrally conditioned rows, border conditioning {border_conditioning} and a \
                 border quotient {border_quotient}"
            ),
            Self::SurrogateLane {
                bytes,
                budget,
                width,
            } => write!(
                f,
                "the exact orbit lane holds {bytes} bytes of arrow blocks and a {width}-square \
                 reduced Schur, over the in-core budget of {budget} bytes, and the rational \
                 surrogate that prices this border carries no orbit pricing (#2234 step 2)"
            ),
            Self::NegativeInertia { count, certificate } => write!(
                f,
                "negative inertia, clamp basin vs saddle undecided on the arrow route: {count} \
                 directions of the stiffened pencil are negative ({certificate})"
            ),
            Self::Band { count, certificate } => write!(
                f,
                "{count} directions of the stiffened pencil lie in the band below its edge \
                 ({certificate}); a band-bearing orbit state is priced on the dense route only \
                 (#2234 step 1b)"
            ),
            Self::Undecided {
                count,
                pivot: Some((resolved, site)),
                certificate,
            } => write!(
                f,
                "undecided: the elimination's weakest pivot, in {site}, is {resolved:.3e} times its \
                 own rounding, so its inertia is not certified, with {count} directions below the \
                 edge ({certificate})"
            ),
            Self::Undecided {
                count,
                pivot: None,
                certificate,
            } => write!(
                f,
                "undecided: {count} directions of the stiffened pencil lie within the certificate's \
                 rounding margin of its band edge ({certificate})"
            ),
            Self::Unresolvable { reason } => write!(f, "unresolvable: {reason}"),
        }
    }
}

/// What the shifted inertia says about the stiffened pencil `(A_s, Φ)`.
#[derive(Clone, Debug)]
pub(crate) enum ArrowOrbitPencilVerdict {
    /// Every direction clears `threshold + margin`.
    Certified(ArrowOrbitCertificate),
    Refused(ArrowOrbitRefusal),
}

/// Classify the stiffened pencil of `operator` along the orbit tangents `tangents` in `metric`,
/// given the value's own elimination at `τ = 0` (#2234 step 1a).
fn certify_orbit_pencil(
    operator: &ArrowJointBlocks,
    metric: &ArrowJointBlocks,
    tangents: ArrayView2<'_, f64>,
    metric_images: ArrayView2<'_, f64>,
    gram: &Array2<f64>,
    gram_inverse: &Array2<f64>,
    metric_frobenius: f64,
    value_factor: &BorderedArrowFactor,
) -> Result<ArrowOrbitPencilVerdict, String> {
    let dim = operator.dim();
    let orbits = tangents.ncols();
    let epsilon = f64::EPSILON;
    let refused = |refusal: ArrowOrbitRefusal| -> Result<ArrowOrbitPencilVerdict, String> {
        Ok(ArrowOrbitPencilVerdict::Refused(refusal))
    };
    let mut operator_images = Array2::<f64>::zeros((dim, orbits));
    for column in 0..orbits {
        operator_images
            .column_mut(column)
            .assign(&operator.apply(tangents.column(column)));
    }
    let projector = metric_images.dot(gram_inverse);
    let middle = tangents.t().dot(&operator_images) + gram;
    let projector_frobenius = frobenius_of(&projector);
    let stiffened_frobenius = operator.frobenius()
        + 2.0 * projector_frobenius * frobenius_of(&operator_images)
        + projector_frobenius * projector_frobenius * frobenius_of(&middle);
    let metric_floor = metric_smallest_eigenvalue_bound(metric)?;
    let coordinate_scale = dim as f64 * epsilon / metric_floor;
    if coordinate_scale * metric_frobenius >= 1.0 {
        return refused(ArrowOrbitRefusal::Unresolvable {
            reason: format!(
                "dim·ε·‖Φ‖_F/λ_min(Φ) = {:.3e} ≥ 1 (dim {dim}, ‖Φ‖_F {metric_frobenius:.3e}, \
                 λ_min(Φ) ≥ {metric_floor:.3e}), so no curvature clears its own resolution",
                coordinate_scale * metric_frobenius
            ),
        });
    }
    let threshold = epsilon
        .sqrt()
        .max(coordinate_scale * stiffened_frobenius / (1.0 - coordinate_scale * metric_frobenius));
    let gamma = dim as f64 * epsilon / (1.0 - dim as f64 * epsilon);
    let margin = gamma
        * (stiffened_frobenius
            + threshold * metric_frobenius
            + 2.0 * frobenius_of(&metric_images))
        / metric_floor;
    let certificate = ArrowOrbitCertificate {
        threshold,
        margin,
        metric_floor,
        stiffened_frobenius,
    };
    let upper = threshold + margin;
    if upper >= 1.0 {
        return refused(ArrowOrbitRefusal::Unresolvable {
            reason: format!(
                "the band edge τ_cert + δ = {upper:.6e} does not clear the orbit's own unit \
                 curvature"
            ),
        });
    }
    let below = |factor: &BorderedArrowFactor| -> Result<usize, String> {
        factor.negative.checked_sub(orbits).ok_or_else(|| {
            format!(
                "arrow orbit lane: the bordered elimination counts {} negative pivots for {orbits} \
                 orbit multipliers, fewer than a full-rank constraint carries",
                factor.negative
            )
        })
    };
    if value_factor.weakest_pivot.0 < 1.0 {
        return refused(ArrowOrbitRefusal::Undecided {
            count: 0,
            pivot: Some(value_factor.weakest_pivot.clone()),
            certificate,
        });
    }
    let negative = below(value_factor)?;
    if negative > 0 {
        return refused(ArrowOrbitRefusal::NegativeInertia {
            count: negative,
            certificate,
        });
    }
    let above = BorderedArrowFactor::eliminate(operator, Some((metric, upper)), metric_images, false)?;
    if above.weakest_pivot.0 < 1.0 {
        return refused(ArrowOrbitRefusal::Undecided {
            count: 0,
            pivot: Some(above.weakest_pivot),
            certificate,
        });
    }
    let below_upper = below(&above)?;
    if below_upper == 0 {
        return Ok(ArrowOrbitPencilVerdict::Certified(certificate));
    }
    let lower = threshold - margin;
    let beneath = BorderedArrowFactor::eliminate(operator, Some((metric, lower)), metric_images, false)?;
    if beneath.weakest_pivot.0 < 1.0 {
        return refused(ArrowOrbitRefusal::Undecided {
            count: below_upper,
            pivot: Some(beneath.weakest_pivot),
            certificate,
        });
    }
    let below_lower = below(&beneath)?;
    if below_lower == 0 {
        refused(ArrowOrbitRefusal::Undecided {
            count: below_upper,
            pivot: None,
            certificate,
        })
    } else {
        refused(ArrowOrbitRefusal::Band {
            count: below_lower,
            certificate,
        })
    }
}

/// #2234 step 1a — one arrow-route evaluation's orbit-eliminated evidence: the value
/// `½log|A_s| − ½log det N − Σ log I_k + ½K·log 2π` the dense route prices, off one elimination of
/// the bordered operator, and the solves its derivative reads. Built only where the stiffened
/// pencil is certified free of band and negative directions.
pub(crate) struct ArrowOrbitGeometry {
    /// The closure-certified circle orbits, in tangent-column order.
    pub(crate) orbit_generators: Vec<CircleOrbitGenerator>,
    total_t: usize,
    k: usize,
    row_offsets: Vec<usize>,
    /// `A`'s arrow blocks, entry for entry the dense materialization's.
    operator: ArrowJointBlocks,
    tangents: Array2<f64>,
    /// `ΦT`.
    metric_images: Array2<f64>,
    gram_inverse: Array2<f64>,
    /// The bordered operator's elimination at `τ = 0`, kept for solves.
    factor: BorderedArrowFactor,
    /// `VN⁻¹`, the off-diagonal block of `K⁻¹`.
    orbit_response: Array2<f64>,
    /// The orbit Schur complement `s·u = σ·N·u`, ascending, as the dense elimination carries it:
    /// its directions `TU`, responses `VU`, their metric images and every `σ`'s band edge.
    curvatures: Array1<f64>,
    directions: Array2<f64>,
    response_directions: Array2<f64>,
    direction_metric_images: Array2<f64>,
    edges: Array1<f64>,
    operator_frobenius: f64,
    pub(crate) certificate: ArrowOrbitCertificate,
    orbits: Vec<CompactOrbitValue>,
    /// `N⁻¹Vᵀu` and `N⁻¹Vᵀv` per orbit: the multiplier part of `K⁻¹[x; 0]`.
    multiplier_images: Vec<(Array1<f64>, Array1<f64>)>,
    /// `log|A_s|`.
    pub(crate) stiffened_log_det: f64,
    /// `−log det N − 2·Σ log I_k + K·log 2π`.
    pub(crate) log_det_correction: f64,
}

impl ArrowOrbitGeometry {
    /// The log-determinant the criterion ranks in place of `log|A|`.
    pub(crate) fn log_det(&self) -> f64 {
        self.stiffened_log_det + self.log_det_correction
    }

    /// `(G_⊥x, N⁻¹Vᵀx)`: the top and multiplier parts of `K⁻¹[x; 0]`.
    fn complement_apply(&self, x: ArrayView1<'_, f64>) -> (Array1<f64>, Array1<f64>) {
        let total_t = self.total_t;
        let orbits = self.tangents.ncols();
        let mut border = Array1::<f64>::zeros(self.k + orbits);
        border
            .slice_mut(s![..self.k])
            .assign(&x.slice(s![total_t..]));
        let (coordinates, reduced) =
            self.factor
                .solve(&self.row_offsets, x.slice(s![..total_t]), border.view());
        let mut complement = Array1::<f64>::zeros(total_t + self.k);
        complement.slice_mut(s![..total_t]).assign(&coordinates);
        complement
            .slice_mut(s![total_t..])
            .assign(&reduced.slice(s![..self.k]));
        (complement, reduced.slice(s![self.k..]).to_owned())
    }

    /// `A⁺·rhs` through the eliminated orbit, as
    /// [`ExactHessianSpectralBlock::solve_orbit_eliminated_stationarity`] forms it: `G_⊥rhs` plus
    /// every resolved orbit curvature's response, certified by the physical residual after removing
    /// its dual components along the in-band orbit directions. A certified complement holds no band.
    pub(crate) fn solve_stationarity(&self, rhs: &SaeArrowVector) -> Result<SaeArrowVector, String> {
        let total_t = self.total_t;
        let dim = total_t + self.k;
        if rhs.t.len() != total_t || rhs.beta.len() != self.k {
            return Err(format!(
                "arrow orbit stationarity solve: RHS ({}, {}) on a ({total_t}, {}) layout",
                rhs.t.len(),
                rhs.beta.len(),
                self.k
            ));
        }
        let mut flat_rhs = Array1::<f64>::zeros(dim);
        flat_rhs.slice_mut(s![..total_t]).assign(&rhs.t);
        flat_rhs.slice_mut(s![total_t..]).assign(&rhs.beta);
        if !flat_rhs.iter().all(|value| value.is_finite()) {
            return Err("arrow orbit stationarity solve: RHS contains a non-finite value".to_string());
        }
        let (mut solution, _) = self.complement_apply(flat_rhs.view());
        for index in 0..self.curvatures.len() {
            if self.curvatures[index].abs() > self.edges[index] {
                let response = self.response_directions.column(index);
                solution.scaled_add(response.dot(&flat_rhs) / self.curvatures[index], &response);
            }
        }
        let residual = &self.operator.apply(solution.view()) - &flat_rhs;
        let mut removed = Array1::<f64>::zeros(dim);
        for index in 0..self.curvatures.len() {
            if self.curvatures[index].abs() <= self.edges[index] {
                removed.scaled_add(
                    self.directions.column(index).dot(&residual),
                    &self.direction_metric_images.column(index),
                );
            }
        }
        let remainder = &residual - &removed;
        let norm = |vector: &Array1<f64>| vector.dot(vector).max(0.0).sqrt();
        let scale = self.operator_frobenius * norm(&solution) + norm(&flat_rhs) + norm(&removed);
        let tolerance = f64::EPSILON.sqrt();
        let remainder_norm = norm(&remainder);
        if !solution.iter().all(|value| value.is_finite())
            || !(remainder_norm == 0.0 || (scale > 0.0 && remainder_norm <= tolerance * scale))
        {
            return Err(format!(
                "arrow orbit stationarity solve failed certification: physical residual off the \
                 held-out orbit directions {remainder_norm:.6e} / backward scale {scale:.6e}, \
                 tolerance {tolerance:.6e}, orbit curvatures {:?} against edges {:?}",
                self.curvatures, self.edges
            ));
        }
        Ok(SaeArrowVector {
            t: solution.slice(s![..total_t]).to_owned(),
            beta: solution.slice(s![total_t..]).to_owned(),
        })
    }
}

/// The differential of the arrow orbit lane's value on its certified stratum, as the dense
/// [`CompactOrbitDifferential`] carries it (full log-determinant units). With every stiffened
/// direction retained, the dense weights reduce to
///
/// ```text
///   dA:  G_⊥,     dΦ:  2·sym(VN⁻¹Tᵀ) − 2·TN⁻¹Tᵀ,     dT:  2Φ·VN⁻¹ − 4Φ·TN⁻¹,
/// ```
///
/// `d log|det K| = ⟨G_⊥, dA⟩ + 2⟨N⁻¹Vᵀ, d(ΦT)⟩` read off `K⁻¹`, less `2·d log det N`, and every leg
/// of `−2·Σ log I_k` is the dense one on the same `G_⊥x` and `N⁻¹Vᵀx`.
struct ArrowOrbitDifferential {
    operator_weight: ArrowJointBlocks,
    metric_weight: ArrowJointBlocks,
    theta: SaeArrowVector,
    log_precisions: Vec<(usize, f64)>,
}

impl SaeManifoldTerm {
    /// #2234 step 1a — the arrow route's orbit-eliminated evidence at `cache`, or the named refusal
    /// of the lane that cannot price it: the ordered Beta--Bernoulli mass and a pinned evidence
    /// factor on the exact lane, a border past the in-core budget on the surrogate lane, and every
    /// state whose stiffened pencil is not certified free of band and negative directions.
    pub(crate) fn arrow_orbit_geometry(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        orbit_generators: Vec<CircleOrbitGenerator>,
    ) -> Result<ArrowOrbitGeometry, SaeCriterionError> {
        let atom = orbit_generators.first().map_or(0, |generator| generator.atom);
        let refuse = |refusal: ArrowOrbitRefusal| {
            SaeCriterionError::OrbitCriterionUnavailableOnArrowRoute { atom, refusal }
        };
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let orbits = orbit_generators.len();
        if orbits == 0 {
            return Err(SaeCriterionError::Numerical(
                "arrow_orbit_geometry: no closure-certified orbit to price".to_string(),
            ));
        }
        if matches!(self.assignment.mode, AssignmentMode::OrderedBetaBernoulli { .. }) {
            return Err(refuse(ArrowOrbitRefusal::CrossRowMass));
        }
        let row_pins: usize = cache.deflated_row_directions.iter().map(Vec::len).sum();
        let conditioned_rows = cache
            .deflation_row_spectra
            .iter()
            .filter(|spectrum| spectrum.is_some())
            .count();
        if cache.gauge_deflated_directions > 0
            || row_pins > 0
            || conditioned_rows > 0
            || cache.beta_schur_conditioning.is_some()
            || cache.beta_gauge_quotient.is_some()
        {
            return Err(refuse(ArrowOrbitRefusal::PinnedEvidenceFactor {
                gauge_directions: cache.gauge_deflated_directions,
                row_pins,
                conditioned_rows,
                border_conditioning: cache.beta_schur_conditioning.is_some(),
                border_quotient: cache.beta_gauge_quotient.is_some(),
            }));
        }
        let width = k + orbits;
        let held = 4usize
            .saturating_mul(total_t.saturating_mul(width))
            .saturating_add(4usize.saturating_mul(width.saturating_mul(width)))
            .saturating_add(dim.saturating_mul(8 * orbits + 8));
        let bytes = held.saturating_mul(std::mem::size_of::<f64>());
        let budget = self.streaming_plan()?.in_core_budget_bytes;
        if bytes > budget {
            return Err(refuse(ArrowOrbitRefusal::SurrogateLane {
                bytes,
                budget,
                width,
            }));
        }
        let mut operator = ArrowJointBlocks::zeros(&cache.row_offsets, k);
        self.probe_exact_hessian_arrow(rho, target, cache, &mut operator)?;
        let metric = ArrowMetric::Joint(cache).prepare()?;
        let metric_blocks = ArrowJointBlocks::from_metric_probes(&cache.row_offsets, k, &metric)?;
        let mut tangents = Array2::<f64>::zeros((dim, orbits));
        let mut metric_images = Array2::<f64>::zeros((dim, orbits));
        for (column, generator) in orbit_generators.iter().enumerate() {
            tangents.column_mut(column).assign(&generator.tangent);
            metric_images
                .column_mut(column)
                .assign(&metric.apply(generator.tangent.view())?);
        }
        let mut gram = tangents.t().dot(&metric_images);
        symmetrized_in_place(&mut gram);
        let gram_inverse = symmetric_positive_function(&gram, "arrow orbit lane", f64::recip)?;
        let metric_frobenius = metric.frobenius_norm()?;
        let factor = BorderedArrowFactor::eliminate(&operator, None, metric_images.view(), true)?;
        let verdict = certify_orbit_pencil(
            &operator,
            &metric_blocks,
            tangents.view(),
            metric_images.view(),
            &gram,
            &gram_inverse,
            metric_frobenius,
            &factor,
        )?;
        let certificate = match verdict {
            ArrowOrbitPencilVerdict::Certified(certificate) => certificate,
            ArrowOrbitPencilVerdict::Refused(refusal) => {
                log::debug!("[SAE-ARROW-ORBIT] atom={atom} orbits={orbits} dim={dim} refused: {refusal}");
                return Err(refuse(refusal));
            }
        };
        // `σ` of the orbit Schur complement: `(K⁻¹)_λλ = −N⁻¹sN⁻¹`.
        let schur_inverse = factor.schur_inverse();
        let multiplier_block = schur_inverse.slice(s![k.., k..]).to_owned();
        let mut orbit_schur = -gram.dot(&multiplier_block).dot(&gram);
        symmetrized_in_place(&mut orbit_schur);
        let root = symmetric_positive_function(&gram, "arrow orbit lane", |value| value.sqrt().recip())?;
        let whitened = root.dot(&orbit_schur).dot(&root);
        let (curvatures, whitened_vectors) = whitened.eigh(Side::Lower).map_err(|error| {
            format!("arrow orbit lane: orbit Schur eigendecomposition failed: {error:?}")
        })?;
        let coordinates = root.dot(&whitened_vectors);
        let mut orbit_response = Array2::<f64>::zeros((dim, orbits));
        for column in 0..orbits {
            let mut border = Array1::<f64>::zeros(width);
            border[k + column] = 1.0;
            let zeros = Array1::<f64>::zeros(total_t);
            let (coordinate_part, border_part) =
                factor.solve(&cache.row_offsets, zeros.view(), border.view());
            orbit_response
                .slice_mut(s![..total_t, column])
                .assign(&coordinate_part);
            orbit_response
                .slice_mut(s![total_t.., column])
                .assign(&border_part.slice(s![..k]));
        }
        let directions = tangents.dot(&coordinates);
        let response_directions = orbit_response.dot(&gram).dot(&coordinates);
        let direction_metric_images = metric_images.dot(&coordinates);
        let operator_frobenius = operator.frobenius();
        let mut edges = Array1::<f64>::zeros(orbits);
        for index in 0..orbits {
            let direction = directions.column(index);
            let resolution = sae_exact_a_pencil_resolution(
                dim,
                direction.dot(&direction),
                operator_frobenius,
                metric_frobenius,
                curvatures[index],
            );
            let substituted = direction
                .dot(&metric.substituted_image(direction)?)
                .max(0.0);
            edges[index] = sae_exact_a_band_edge(curvatures[index], resolution, substituted);
        }
        let (gram_values, _) = gram.eigh(Side::Lower).map_err(|error| {
            format!("arrow orbit lane: Gram eigendecomposition failed: {error:?}")
        })?;
        let log_gram_det = gram_values.iter().map(|value| value.ln()).sum::<f64>();
        let stiffened_log_det = factor.log_abs_det - log_gram_det;
        let mut geometry = ArrowOrbitGeometry {
            orbit_generators,
            total_t,
            k,
            row_offsets: cache.row_offsets.to_vec(),
            operator,
            tangents,
            metric_images,
            gram_inverse,
            factor,
            orbit_response,
            curvatures,
            directions,
            response_directions,
            direction_metric_images,
            edges,
            operator_frobenius,
            certificate,
            orbits: Vec::with_capacity(orbits),
            multiplier_images: Vec::with_capacity(orbits),
            stiffened_log_det,
            log_det_correction: 0.0,
        };
        let mut log_integrals = 0.0_f64;
        for index in 0..orbits {
            let (u, v) = geometry.orbit_generators[index].trigonometric_images(dim);
            let (complement_u, multiplier_u) = geometry.complement_apply(u.view());
            let (complement_v, multiplier_v) = geometry.complement_apply(v.view());
            let coupling_forms = [
                u.dot(&complement_u),
                0.5 * (u.dot(&complement_v) + v.dot(&complement_u)),
                v.dot(&complement_v),
            ];
            let integral = geometry.orbit_generators[index]
                .integrand(coupling_forms[0], coupling_forms[1], coupling_forms[2])
                .integrate()?;
            log::debug!(
                "[SAE-ARROW-ORBIT] atom={} priced: nodes={} log I={:.6e} log det N={:.6e} \
                 coupling=[{:.3e}, {:.3e}, {:.3e}]",
                geometry.orbit_generators[index].atom,
                integral.angles.len(),
                integral.log_integral,
                log_gram_det,
                coupling_forms[0],
                coupling_forms[1],
                coupling_forms[2],
            );
            log_integrals += integral.log_integral;
            geometry.orbits.push(CompactOrbitValue {
                trigonometric: (u, v),
                complement_images: (complement_u, complement_v),
                coupling_forms,
                integral,
            });
            geometry.multiplier_images.push((multiplier_u, multiplier_v));
        }
        geometry.log_det_correction =
            -log_gram_det - 2.0 * log_integrals + orbits as f64 * std::f64::consts::TAU.ln();
        log::debug!(
            "[SAE-ARROW-ORBIT] certified: τ_cert={:.6e} δ={:.3e} λ_min(Φ)≥{:.3e} ½log|A_s|={:.6e} \
             correction={:.6e} orbit curvatures {:?} against edges {:?}",
            geometry.certificate.threshold,
            geometry.certificate.margin,
            geometry.certificate.metric_floor,
            0.5 * geometry.stiffened_log_det,
            geometry.log_det_correction,
            geometry.curvatures,
            geometry.edges,
        );
        Ok(geometry)
    }

    fn arrow_orbit_differential(
        geometry: &ArrowOrbitGeometry,
        metric: &dyn ExactAPencilMetric,
    ) -> Result<ArrowOrbitDifferential, String> {
        let (total_t, k) = (geometry.total_t, geometry.k);
        let dim = total_t + k;
        let orbits = geometry.tangents.ncols();
        let tangents = &geometry.tangents;
        let gram_inverse = &geometry.gram_inverse;
        // `G_⊥` on the arrow positions: the top-left block of `K⁻¹ = D̂⁻¹ + E S⁻¹ Eᵀ`,
        // `E = [−F; I]`.
        let schur_inverse = geometry.factor.schur_inverse();
        let border_columns = schur_inverse.slice(s![.., ..k]).to_owned();
        let mut operator_weight = ArrowJointBlocks::zeros(&geometry.row_offsets, k);
        for (row, (values, vectors)) in geometry.factor.rows.iter().enumerate() {
            let (start, end) = (geometry.row_offsets[row], geometry.row_offsets[row + 1]);
            if start == end {
                continue;
            }
            let coupled = geometry.factor.coupled.slice(s![start..end, ..]);
            let scaled = Array2::from_shape_fn(vectors.raw_dim(), |(i, j)| vectors[[i, j]] / values[j]);
            let local_inverse = scaled.dot(&vectors.t());
            operator_weight.rows[row] = local_inverse + coupled.dot(&schur_inverse).dot(&coupled.t());
            operator_weight
                .cross
                .slice_mut(s![start..end, ..])
                .assign(&(-coupled.dot(&border_columns)));
        }
        operator_weight
            .border
            .assign(&schur_inverse.slice(s![..k, ..k]));
        let lifted_tangents = tangents.dot(gram_inverse);
        let mut metric_weight = ArrowJointBlocks::zeros(&geometry.row_offsets, k);
        for column in 0..orbits {
            metric_weight.add_symmetric_outer(
                2.0,
                geometry.orbit_response.column(column),
                tangents.column(column),
            );
            metric_weight.add_symmetric_outer(
                -2.0,
                tangents.column(column),
                lifted_tangents.column(column),
            );
        }
        let metric_lifted = geometry.metric_images.dot(gram_inverse);
        let mut tangent_weight = Array2::<f64>::zeros((dim, orbits));
        for column in 0..orbits {
            let image = metric.apply(geometry.orbit_response.column(column))?;
            let mut target = tangent_weight.column_mut(column);
            target.scaled_add(2.0, &image);
            target.scaled_add(-4.0, &metric_lifted.column(column));
        }

        // −2·Σ log I_k, leg for leg as `compact_orbit_differential` prices it.
        let one_minus_cos = |angle: f64| {
            let half = (0.5 * angle).sin();
            2.0 * half * half
        };
        let mut theta = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(k),
        };
        let mut log_precisions = Vec::with_capacity(orbits);
        for ((generator, value), (multiplier_u, multiplier_v)) in geometry
            .orbit_generators
            .iter()
            .zip(geometry.orbits.iter())
            .zip(geometry.multiplier_images.iter())
        {
            let (complement_u, complement_v) = &value.complement_images;
            let angles = &value.integral.angles;
            let node_weights = &value.integral.weights;
            let expect = |function: &dyn Fn(f64) -> f64| -> f64 {
                angles
                    .iter()
                    .zip(node_weights.iter())
                    .map(|(&angle, &node_weight)| node_weight * function(angle))
                    .sum()
            };
            let eta = generator.eta;
            let kappa = generator.kappa;
            let coupling_scale = 0.5 * eta * eta * kappa * kappa;
            let mean_one_minus_cos = expect(&one_minus_cos);
            let mean_sin = expect(&|angle: f64| angle.sin());
            let weight_a = coupling_scale * expect(&|angle: f64| one_minus_cos(angle).powi(2));
            let weight_b =
                -2.0 * coupling_scale * expect(&|angle: f64| one_minus_cos(angle) * angle.sin());
            let weight_d = coupling_scale * expect(&|angle: f64| angle.sin().powi(2));
            // `(weight, G_⊥x, G_⊥y, N⁻¹Vᵀx, N⁻¹Vᵀy)` of each coupling form `xᵀG_⊥y`.
            let forms: [(f64, &Array1<f64>, &Array1<f64>, &Array1<f64>, &Array1<f64>); 3] = [
                (weight_a, complement_u, complement_u, multiplier_u, multiplier_u),
                (weight_b, complement_u, complement_v, multiplier_u, multiplier_v),
                (weight_d, complement_v, complement_v, multiplier_v, multiplier_v),
            ];
            for (form_weight, complement_x, complement_y, coefficient_x, coefficient_y) in forms {
                if form_weight == 0.0 {
                    continue;
                }
                // d(−2·log I) = −2·(∂log I/∂f)·df, and every leg of df carries a leading minus.
                let scale = 2.0 * form_weight;
                operator_weight.add_symmetric_outer(scale, complement_x.view(), complement_y.view());
                let lift_x = tangents.dot(coefficient_x);
                let lift_y = tangents.dot(coefficient_y);
                metric_weight.add_symmetric_outer(scale, complement_x.view(), lift_y.view());
                metric_weight.add_symmetric_outer(scale, lift_x.view(), complement_y.view());
                let metric_complement_x = metric.apply(complement_x.view())?;
                let metric_complement_y = metric.apply(complement_y.view())?;
                for column in 0..orbits {
                    let mut target = tangent_weight.column_mut(column);
                    target.scaled_add(scale * coefficient_y[column], &metric_complement_x);
                    target.scaled_add(scale * coefficient_x[column], &metric_complement_y);
                }
            }
            for &(slot, row_weight, coordinate) in &generator.prior_rows {
                let (sin, cos) = (kappa * coordinate).sin_cos();
                let resultant = -eta * kappa * row_weight * (-sin * mean_one_minus_cos + cos * mean_sin);
                let moved_u = row_weight * kappa * cos;
                let moved_v = -row_weight * kappa * sin;
                let coupling = weight_a * 2.0 * complement_u[slot] * moved_u
                    + weight_b * (complement_v[slot] * moved_u + complement_u[slot] * moved_v)
                    + weight_d * 2.0 * complement_v[slot] * moved_v;
                theta.t[slot] -= 2.0 * (resultant + coupling);
            }
            let integrand =
                generator.integrand(value.coupling_forms[0], value.coupling_forms[1], value.coupling_forms[2]);
            let [qa, qb, qd] = integrand.coupling;
            let mean_ard =
                -(integrand.resultant_cos * mean_one_minus_cos + integrand.resultant_sin * mean_sin);
            let mean_coupling = expect(&|angle: f64| {
                let lowered = one_minus_cos(angle);
                let sin = angle.sin();
                qa * lowered * lowered - 2.0 * qb * lowered * sin + qd * sin * sin
            });
            log_precisions.push((generator.atom, -2.0 * (mean_ard + 2.0 * mean_coupling)));
        }
        // Each tangent's border block is −K̃C on its own atom, so its column's weight reaches that
        // atom's decoder coordinates as −K̃ᵀ.
        for (column_of_tangent, generator) in geometry.orbit_generators.iter().enumerate() {
            let rank = generator.border_rank;
            for column in 0..generator.basis_size {
                for channel in 0..rank {
                    let moved =
                        tangent_weight[[generator.border_start + column * rank + channel, column_of_tangent]];
                    if moved == 0.0 {
                        continue;
                    }
                    for j in 0..generator.basis_size {
                        theta.beta[generator.border_start - total_t + j * rank + channel] -=
                            generator.closure[[column, j]] * moved;
                    }
                }
            }
        }
        Ok(ArrowOrbitDifferential {
            operator_weight,
            metric_weight,
            theta,
            log_precisions,
        })
    }

    /// #2234 step 1a — the arrow orbit lane's log-determinant channels, as
    /// [`Self::dense_exact_a_logdet_channels`] prices them on the dense route: `½⟨W_A, ∂A/∂ρ⟩` and
    /// the θ-adjoint of the `dA` weight, the evidence factor's own `dΦ` legs (an unpinned factor,
    /// so the weight on `dB_raw` is the weight on `dΦ`), the orbit legs that reach neither
    /// operator, and the stationarity adjoint through the eliminated orbit. A certified state
    /// retains every stiffened direction, so it carries no band or basin weight.
    pub(crate) fn arrow_orbit_logdet_channels(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
        geometry: &ArrowOrbitGeometry,
        rank_charge_theta: &SaeArrowVector,
    ) -> Result<DenseExactALogdetChannels, String> {
        let total_t = cache.delta_t_len();
        if geometry.total_t != total_t || geometry.k != cache.k {
            return Err(format!(
                "arrow_orbit_logdet_channels: the geometry ({}, {}) was not built at this cache \
                 ({total_t}, {})",
                geometry.total_t, geometry.k, cache.k
            ));
        }
        let metric = ArrowMetric::Joint(cache).prepare()?;
        let mut differential = Self::arrow_orbit_differential(geometry, &metric)?;
        // #3439 — the periodic phases keep their circle volume beside the orbit's, priced
        // off this cache as the value priced it. Its weight is on `dB_raw`, which on this
        // unpinned factor is the weight on `dΦ`, so it joins the metric weight.
        if let Some((_, phase)) = self.periodic_phase_marginal(cache)? {
            differential.metric_weight.accumulate(&phase)?;
        }
        let mut logdet_trace = Array1::<f64>::zeros(rho.flat_coordinates().len());
        let mut operator_traces = ContractingPenaltyDerivatives::new(&differential.operator_weight);
        self.raw_penalty_curvature_operators_into(rho, cache, &mut operator_traces)?;
        self.exact_stationarity_penalty_derivative_delta_into(rho, cache, &mut operator_traces)?;
        for (flat, contraction) in operator_traces.contractions {
            logdet_trace[flat] = 0.5 * contraction;
        }
        // #2231 — the crosscoder block weights reach `A` through the scaled target.
        self.add_crosscoder_block_logdet_traces(
            rho,
            target,
            cache,
            &differential.operator_weight,
            &mut logdet_trace,
        )?;
        let mut gamma = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &differential.operator_weight,
            true,
            true,
            Some(target),
        )?;
        let mut metric_traces = ContractingPenaltyDerivatives::new(&differential.metric_weight);
        self.raw_penalty_curvature_operators_into(rho, cache, &mut metric_traces)?;
        for (flat, contraction) in metric_traces.contractions {
            logdet_trace[flat] += 0.5 * contraction;
        }
        let metric_gamma = self.logdet_theta_adjoint_dense(
            rho,
            cache,
            &differential.metric_weight,
            true,
            false,
            Some(target),
        )?;
        gamma.t += &metric_gamma.t;
        gamma.beta += &metric_gamma.beta;
        if cache.k > 0 {
            // `B_ββ` carries the decoder priors' majorizer, `A_ββ + E_ββ`.
            let border = differential.metric_weight.border_block(total_t).ok_or_else(|| {
                "arrow_orbit_logdet_channels: the metric weight holds no border at this cache".to_string()
            })?;
            gamma.beta += &self.exact_decoder_prior_theta_trace(cache, border)?;
            gamma.beta += &self.decoder_prior_gap_theta_trace(cache, border)?;
        }
        gamma.t += &differential.theta.t;
        gamma.beta += &differential.theta.beta;
        for (atom, log_precision) in differential.log_precisions {
            if !rho.log_ard[atom].is_empty() {
                logdet_trace[rho.ard_flat_index(atom, 0)] += 0.5 * log_precision;
            }
        }
        gamma.t.scaled_add(2.0, &rank_charge_theta.t);
        gamma.beta.scaled_add(2.0, &rank_charge_theta.beta);
        let stationarity_adjoint = geometry.solve_stationarity(&gamma)?;
        Ok(DenseExactALogdetChannels {
            logdet_trace,
            theta_adjoint: gamma,
            stationarity_adjoint,
        })
    }
}

//! Tiled resident attention using the existing tensor backend. The fast full-matrix route is
//! retained for small batches; this route bounds sequence-square scratch in larger workloads.
use gam_gpu::{gpu_error::GpuError, tensor::{Arithmetic, Device, Op, Tensor}};
use std::ops::Range;
const TILE: usize = 256;

/// A run of a call's rows that is one sequence's positions `first..first + rows.len()`, whose
/// earlier positions' keys and values are the call's rows `before` (in position order): another
/// sequence's rows where the two agree before `first` (`interchange`'s suffix lanes), so the
/// attention of the run's queries reads `before` and then its own rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub rows: Range<usize>,
    pub first: usize,
    pub before: Vec<Range<usize>>,
}

impl Segment {
    /// The call's rows holding the segment's keys and values, in position order.
    pub(crate) fn keys(&self) -> Vec<Range<usize>> {
        let mut keys = self.before.clone();
        keys.push(self.rows.clone());
        keys
    }

    /// The keys' count, `first` plus the segment's rows.
    pub(crate) fn length(&self) -> usize {
        self.first + self.rows.len()
    }
}

/// The length every segment's keys share, when they do: a call of whole sequences and suffix lanes
/// of one sequence length (every run reaches its sequence's end). The segments then run as that many
/// sequences at once ([`Whole`]).
pub(crate) fn common_length(segments: &[Segment]) -> Option<usize> {
    let length = segments.first()?.length();
    segments.iter().all(|s| s.length() == length).then_some(length)
}

/// Whether the attention of sequences of `length` in `rows` runs in query tiles rather than with
/// every sequence's weights at once: a sequence longer than 1024 positions or more than 8M weights.
pub(crate) fn tiles(rows: usize, length: usize) -> bool {
    length > 1024 || rows.saturating_mul(length) > 8 * 1024 * 1024
}

/// Segments of one length as whole sequences: their keys' rows gathered one sequence after another
/// (`keys`), each segment's own rows there (`own`, after its earlier positions) and in the call
/// (`rows`), and the earlier positions' rows there (`earlier`) with the call's rows holding them.
pub(crate) struct Whole {
    pub(crate) keys: Vec<Range<usize>>,
    pub(crate) own: Vec<Range<usize>>,
    pub(crate) rows: Vec<Range<usize>>,
    pub(crate) earlier: Vec<(Range<usize>, Vec<Range<usize>>)>,
}

impl Whole {
    pub(crate) fn of(segments: &[Segment], length: usize) -> Self {
        let keys = segments.iter().flat_map(Segment::keys).collect();
        let own = segments.iter().enumerate().map(|(i, s)| i * length + s.first..(i + 1) * length).collect();
        let rows = segments.iter().map(|s| s.rows.clone()).collect();
        let earlier = segments.iter().enumerate().filter(|(_, s)| s.first > 0).map(|(i, s)| (i * length..i * length + s.first, s.before.clone())).collect();
        Self { keys, own, rows, earlier }
    }

    /// The whole sequences' outputs `all` at the call's rows (`rows` of them): each segment's own.
    pub(crate) fn outputs(&self, d: &Device, all: &Tensor, rows: usize) -> Result<Tensor, GpuError> {
        let mut out = d.zeros(rows, all.cols())?;
        d.scatter_ranges(&mut out, &self.rows, &d.gather_ranges(all, &self.own)?)?;
        Ok(out)
    }

    /// A cotangent of the call's rows as one of the whole sequences': each segment's own rows'
    /// where its own rows are there, zero at its earlier positions (their queries are its twin's).
    pub(crate) fn spread(&self, d: &Device, cot: &Tensor) -> Result<Tensor, GpuError> {
        let mut all = d.zeros(self.keys.iter().map(ExactSizeIterator::len).sum(), cot.cols())?;
        d.scatter_ranges(&mut all, &self.own, &d.gather_ranges(cot, &self.rows)?)?;
        Ok(all)
    }

    /// The whole sequences' cotangent `all` of their keys' rows at the call's rows (`rows` of them):
    /// each segment's own rows' as they are, its earlier positions' added into the rows holding
    /// them (another segment's own rows, segment by segment).
    pub(crate) fn gather_back(&self, d: &Device, all: &Tensor, rows: usize) -> Result<Tensor, GpuError> {
        let mut out = d.zeros(rows, all.cols())?;
        d.scatter_ranges(&mut out, &self.rows, &d.gather_ranges(all, &self.own)?)?;
        for (at, before) in &self.earlier {
            add_rows(d, &mut out, before, &d.rows_of(all, at.start, at.len())?)?;
        }
        Ok(out)
    }
}

/// The attention of `blocks` sequences with every sequence's weights at once (the full-matrix
/// route), its products in `arithmetic`.
fn forward_whole(d: &Device, (q, k, v): Values<'_>, blocks: usize, scale: f64, causal: bool, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let length = q.rows() / blocks;
    let mut alpha = d.empty(q.rows(), length)?;
    d.gemm_batched(blocks, &mut alpha, scale, q, Op::N, k, Op::T, 0.0, arithmetic)?;
    d.softmax_rows(&mut alpha, causal)?;
    let mut out = d.empty(q.rows(), v.cols())?;
    d.gemm_batched(blocks, &mut out, 1.0, &alpha, Op::N, v, Op::N, 0.0, arithmetic)?;
    Ok(out)
}

/// [`forward_whole`]'s cotangents of the queries, keys and values: the weights again in the
/// forward's arithmetic `forward`, the products in `arithmetic`.
fn backward_whole(d: &Device, (q, k, v): Values<'_>, cot: &Tensor, blocks: usize, scale: f64, causal: bool, (forward, arithmetic): (Arithmetic, Arithmetic)) -> Result<Triple, GpuError> {
    let length = q.rows() / blocks;
    let mut alpha = d.empty(q.rows(), length)?;
    d.gemm_batched(blocks, &mut alpha, scale, q, Op::N, k, Op::T, 0.0, forward)?;
    d.softmax_rows(&mut alpha, causal)?;
    let mut dalpha = d.empty(q.rows(), length)?;
    d.gemm_batched(blocks, &mut dalpha, 1.0, cot, Op::N, v, Op::T, 0.0, arithmetic)?;
    let mut gv = d.empty(v.rows(), v.cols())?;
    d.gemm_batched(blocks, &mut gv, 1.0, &alpha, Op::T, cot, Op::N, 0.0, arithmetic)?;
    let ds = d.softmax_backward(&alpha, &dalpha)?;
    drop((alpha, dalpha));
    let mut gq = d.empty(q.rows(), q.cols())?;
    d.gemm_batched(blocks, &mut gq, scale, &ds, Op::N, k, Op::N, 0.0, arithmetic)?;
    let mut gk = d.empty(k.rows(), k.cols())?;
    d.gemm_batched(blocks, &mut gk, scale, &ds, Op::T, q, Op::N, 0.0, arithmetic)?;
    Ok((gq, gk, gv))
}

/// `values` added into `target`'s rows `ranges` (in order; ranges of several segments may share
/// rows, each segment's addition in turn).
pub(crate) fn add_rows(d: &Device, target: &mut Tensor, ranges: &[Range<usize>], values: &Tensor) -> Result<(), GpuError> {
    let mut rows = d.gather_ranges(target, ranges)?;
    d.axpy(&mut rows, 1.0, values)?;
    d.scatter_ranges(target, ranges, &rows)
}

/// [`forward`] over `segments`: each segment's queries against its keys and values, the earlier
/// positions' from its `before` rows.
pub(crate) fn forward_segments(d: &Device, (q, k, v): Values<'_>, segments: &[Segment], scale: f64, causal: bool, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    // Segments of one length run as whole sequences at once where their weights fit (`tiles`): a
    // few launches for the call instead of several for each segment's every query tile.
    if let Some(length) = common_length(segments)
        && !tiles(segments.len() * length, length)
    {
        let whole = Whole::of(segments, length);
        let (qs, ks, vs) = (d.gather_ranges(q, &whole.keys)?, d.gather_ranges(k, &whole.keys)?, d.gather_ranges(v, &whole.keys)?);
        let all = forward_whole(d, (&qs, &ks, &vs), segments.len(), scale, causal, arithmetic)?;
        return whole.outputs(d, &all, q.rows());
    }
    let mut out = d.zeros(q.rows(), v.cols())?;
    for s in segments {
        let keys = s.keys();
        let (ks, vs) = (d.gather_ranges(k, &keys)?, d.gather_ranges(v, &keys)?);
        let n = s.rows.len();
        for start in (0..n).step_by(TILE) {
            let m = TILE.min(n - start);
            let qb = d.rows_of(q, s.rows.start + start, m)?;
            let p = probabilities(d, &qb, &ks, s.first + start, scale, causal, arithmetic)?;
            let mut tile = d.zeros(m, v.cols())?;
            d.gemm(&mut tile, 1.0, &p, Op::N, &vs, Op::N, 0.0, arithmetic)?;
            d.set_rows(&mut out, s.rows.start + start, &tile)?;
        }
    }
    Ok(out)
}

/// [`backward`] over `segments`: the keys' and values' cotangents of every segment added into
/// the rows holding them (its own and its `before` rows).
pub(crate) fn backward_segments(d: &Device, (q, k, v): Values<'_>, cot: &Tensor, segments: &[Segment], scale: f64, causal: bool, (forward, arithmetic): (Arithmetic, Arithmetic)) -> Result<Triple, GpuError> {
    // As `forward_segments`: whole sequences at once, the cotangent on each segment's own rows; the
    // queries' cotangent at a segment's earlier positions is zero (no output there is its own).
    if let Some(length) = common_length(segments)
        && !tiles(segments.len() * length, length)
    {
        let whole = Whole::of(segments, length);
        let (qs, ks, vs) = (d.gather_ranges(q, &whole.keys)?, d.gather_ranges(k, &whole.keys)?, d.gather_ranges(v, &whole.keys)?);
        let (gq, gk, gv) = backward_whole(d, (&qs, &ks, &vs), &whole.spread(d, cot)?, segments.len(), scale, causal, (forward, arithmetic))?;
        return Ok((whole.outputs(d, &gq, q.rows())?, whole.gather_back(d, &gk, k.rows())?, whole.gather_back(d, &gv, v.rows())?));
    }
    let (mut gq, mut gk, mut gv) = (d.zeros(q.rows(), q.cols())?, d.zeros(k.rows(), k.cols())?, d.zeros(v.rows(), v.cols())?);
    for s in segments {
        let keys = s.keys();
        let (ks, vs) = (d.gather_ranges(k, &keys)?, d.gather_ranges(v, &keys)?);
        let (length, n) = (s.length(), s.rows.len());
        let (mut key_grad, mut value_grad) = (d.zeros(length, k.cols())?, d.zeros(length, v.cols())?);
        for start in (0..n).step_by(TILE) {
            let m = TILE.min(n - start);
            let qb = d.rows_of(q, s.rows.start + start, m)?;
            let cb = d.rows_of(cot, s.rows.start + start, m)?;
            let p = probabilities(d, &qb, &ks, s.first + start, scale, causal, forward)?;
            let mut dp = d.zeros(m, length)?;
            d.gemm(&mut dp, 1.0, &cb, Op::N, &vs, Op::T, 0.0, arithmetic)?;
            let ds = d.softmax_backward(&p, &dp)?;
            drop(dp);
            let mut query_grad = d.zeros(m, q.cols())?;
            d.gemm(&mut query_grad, scale, &ds, Op::N, &ks, Op::N, 0.0, arithmetic)?;
            d.gemm(&mut key_grad, scale, &ds, Op::T, &qb, Op::N, 1.0, arithmetic)?;
            d.gemm(&mut value_grad, 1.0, &p, Op::T, &cb, Op::N, 1.0, arithmetic)?;
            d.set_rows(&mut gq, s.rows.start + start, &query_grad)?;
        }
        add_rows(d, &mut gk, &keys, &key_grad)?;
        add_rows(d, &mut gv, &keys, &value_grad)?;
    }
    Ok((gq, gk, gv))
}

/// [`tangent`] over `segments`.
pub(crate) fn tangent_segments(d: &Device, (q, k, v): Values<'_>, (dq, dk, dv): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>), segments: &[Segment], scale: f64, causal: bool, (forward, arithmetic): (Arithmetic, Arithmetic)) -> Result<Tensor, GpuError> {
    let mut out = d.zeros(q.rows(), v.cols())?;
    for s in segments {
        let keys = s.keys();
        let (ks, vs) = (d.gather_ranges(k, &keys)?, d.gather_ranges(v, &keys)?);
        let dks = dk.map(|t| d.gather_ranges(t, &keys)).transpose()?;
        let dvs = dv.map(|t| d.gather_ranges(t, &keys)).transpose()?;
        let (length, n) = (s.length(), s.rows.len());
        for start in (0..n).step_by(TILE) {
            let m = TILE.min(n - start);
            let qb = d.rows_of(q, s.rows.start + start, m)?;
            let p = probabilities(d, &qb, &ks, s.first + start, scale, causal, forward)?;
            let mut ds = d.zeros(m, length)?;
            if let Some(dq) = dq {
                let dqb = d.rows_of(dq, s.rows.start + start, m)?;
                d.gemm(&mut ds, scale, &dqb, Op::N, &ks, Op::T, 1.0, arithmetic)?;
            }
            if let Some(dks) = &dks {
                d.gemm(&mut ds, scale, &qb, Op::N, dks, Op::T, 1.0, arithmetic)?;
            }
            let dp = d.softmax_backward(&p, &ds)?;
            drop(ds);
            let mut tile = d.zeros(m, v.cols())?;
            d.gemm(&mut tile, 1.0, &dp, Op::N, &vs, Op::N, 0.0, arithmetic)?;
            if let Some(dvs) = &dvs {
                d.gemm(&mut tile, 1.0, &p, Op::N, dvs, Op::N, 1.0, arithmetic)?;
            }
            d.set_rows(&mut out, s.rows.start + start, &tile)?;
        }
    }
    Ok(out)
}
type Triple = (Tensor, Tensor, Tensor);
type Values<'a> = (&'a Tensor, &'a Tensor, &'a Tensor);

/// The weights of a query tile, their product in the forward pass's `arithmetic`.
fn probabilities(d: &Device, q: &Tensor, k: &Tensor, start: usize, scale: f64, causal: bool, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let mut p = d.zeros(q.rows(), k.rows())?;
    d.gemm(&mut p, scale, q, Op::N, k, Op::T, 0.0, arithmetic)?;
    d.softmax_rows_offset(&mut p, causal, start)?;
    Ok(p)
}

/// The attention of `blocks` sequences, its products in `arithmetic`.
pub(crate) fn forward(d: &Device, (q, k, v): Values<'_>, blocks: usize, scale: f64, causal: bool, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
    let length = q.rows() / blocks;
    let mut out = d.zeros(q.rows(), v.cols())?;
    for block in 0..blocks {
        let base = block * length;
        let ks = d.rows_of(k, base, length)?;
        let vs = d.rows_of(v, base, length)?;
        for start in (0..length).step_by(TILE) {
            let n = TILE.min(length - start);
            let qb = d.rows_of(q, base + start, n)?;
            let p = probabilities(d, &qb, &ks, start, scale, causal, arithmetic)?;
            let mut tile = d.zeros(n, v.cols())?;
            d.gemm(&mut tile, 1.0, &p, Op::N, &vs, Op::N, 0.0, arithmetic)?;
            d.set_rows(&mut out, base + start, &tile)?;
        }
    }
    Ok(out)
}

/// The cotangents of [`forward`]'s inputs, its products in `arithmetic` and the weights
/// recomputed in the forward pass's, `forward`.
pub(crate) fn backward(d: &Device, (q, k, v): Values<'_>, cot: &Tensor, blocks: usize, scale: f64, causal: bool, (forward, arithmetic): (Arithmetic, Arithmetic)) -> Result<Triple, GpuError> {
    let length = q.rows() / blocks;
    let (mut gq, mut gk, mut gv) = (d.zeros(q.rows(), q.cols())?, d.zeros(k.rows(), k.cols())?, d.zeros(v.rows(), v.cols())?);
    for block in 0..blocks {
        let base = block * length;
        let ks = d.rows_of(k, base, length)?;
        let vs = d.rows_of(v, base, length)?;
        let (mut key_grad, mut value_grad) = (d.zeros(length, k.cols())?, d.zeros(length, v.cols())?);
        for start in (0..length).step_by(TILE) {
            let n = TILE.min(length - start);
            let qb = d.rows_of(q, base + start, n)?;
            let cb = d.rows_of(cot, base + start, n)?;
            let p = probabilities(d, &qb, &ks, start, scale, causal, forward)?;
            let mut dp = d.zeros(n, length)?;
            d.gemm(&mut dp, 1.0, &cb, Op::N, &vs, Op::T, 0.0, arithmetic)?;
            let ds = d.softmax_backward(&p, &dp)?;
            drop(dp);
            let mut query_grad = d.zeros(n, q.cols())?;
            d.gemm(&mut query_grad, scale, &ds, Op::N, &ks, Op::N, 0.0, arithmetic)?;
            d.gemm(&mut key_grad, scale, &ds, Op::T, &qb, Op::N, 1.0, arithmetic)?;
            d.gemm(&mut value_grad, 1.0, &p, Op::T, &cb, Op::N, 1.0, arithmetic)?;
            d.set_rows(&mut gq, base + start, &query_grad)?;
        }
        d.set_rows(&mut gk, base, &key_grad)?;
        d.set_rows(&mut gv, base, &value_grad)?;
    }
    Ok((gq, gk, gv))
}

/// The tangent of [`forward`] along `(dq, dk, dv)`, as [`backward`] takes its arithmetic.
pub(crate) fn tangent(d: &Device, (q, k, v): Values<'_>, (dq, dk, dv): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>), blocks: usize, scale: f64, causal: bool, (forward, arithmetic): (Arithmetic, Arithmetic)) -> Result<Tensor, GpuError> {
    let length = q.rows() / blocks;
    let mut out = d.zeros(q.rows(), v.cols())?;
    for block in 0..blocks {
        let base = block * length;
        let ks = d.rows_of(k, base, length)?;
        let vs = d.rows_of(v, base, length)?;
        let dks = dk.map(|t| d.rows_of(t, base, length)).transpose()?;
        let dvs = dv.map(|t| d.rows_of(t, base, length)).transpose()?;
        for start in (0..length).step_by(TILE) {
            let n = TILE.min(length - start);
            let qb = d.rows_of(q, base + start, n)?;
            let p = probabilities(d, &qb, &ks, start, scale, causal, forward)?;
            let mut ds = d.zeros(n, length)?;
            if let Some(dq) = dq {
                let dqb = d.rows_of(dq, base + start, n)?;
                d.gemm(&mut ds, scale, &dqb, Op::N, &ks, Op::T, 1.0, arithmetic)?;
            }
            if let Some(dks) = &dks { d.gemm(&mut ds, scale, &qb, Op::N, dks, Op::T, 1.0, arithmetic)?; }
            let dp = d.softmax_backward(&p, &ds)?;
            drop(ds);
            let mut tile = d.zeros(n, v.cols())?;
            d.gemm(&mut tile, 1.0, &dp, Op::N, &vs, Op::N, 0.0, arithmetic)?;
            if let Some(dvs) = &dvs { d.gemm(&mut tile, 1.0, &p, Op::N, dvs, Op::N, 1.0, arithmetic)?; }
            d.set_rows(&mut out, base + start, &tile)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Segments against whole sequences: two sequences of 300 rows equal before row 77, laid out as
    /// the first whole (call rows 0..300) and the second from row 77 on (call rows 300..523, its
    /// keys before 77 the first's rows 0..77), and a third whole sequence (call rows 523..823). Each
    /// segment's outputs equal whole-sequence attention's at the same rows, and its cotangents and
    /// tangents those of the whole-sequence pass with the second sequence's rows before 77 added to
    /// the first's (the rows the segment reads in their place), across tile boundaries.
    #[test]
    fn segments_read_their_earlier_rows_from_another_sequence() {
        let (length, first, width, values) = (300, 77, 8, 5);
        let data = |rows: usize, w: usize, salt: usize| Array2::from_shape_fn((rows, w), |(r, c)| ((r * 31 + c * 7 + salt) as f64 * 0.73).sin());
        // Whole sequences 0, 1, 2 (1 equal to 0 before `first`).
        let whole = |w: usize, salt: usize| {
            let mut x = data(3 * length, w, salt);
            for r in 0..first {
                let row = x.row(r).to_owned();
                x.row_mut(length + r).assign(&row);
            }
            x
        };
        let (q, k, v, dq, dk, dv, cot) = (whole(width, 1), whole(width, 2), whole(values, 3), whole(width, 4), whole(width, 5), whole(values, 6), data(3 * length, values, 7));
        let mut cot = cot;
        for r in 0..first {
            cot.row_mut(length + r).fill(0.0);
        }
        // The call's rows: sequence 0, sequence 1 from `first` on, sequence 2.
        let kept: Vec<usize> = (0..length).chain(length + first..2 * length).chain(2 * length..3 * length).collect();
        let call = |x: &Array2<f64>| x.select(ndarray::Axis(0), &kept);
        let segments = vec![
            Segment { rows: 0..length, first: 0, before: vec![] },
            Segment { rows: length..2 * length - first, first, before: vec![0..first] },
            Segment { rows: 2 * length - first..3 * length - first, first: 0, before: vec![] },
        ];
        for d in crate::device_program_tests::devices() {
            let up = |x: &Array2<f64>| d.upload(x.view()).expect("upload");
            let close = |what: &str, actual: &Tensor, expected: &Array2<f64>| {
                let actual = d.download(actual).expect("download");
                let error = (&actual - expected).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                assert!(error < 1e-10, "{}: {what} max error {error}", d.name());
            };
            let (qw, kw, vw, dqw, dkw, dvw, cw) = (up(&q), up(&k), up(&v), up(&dq), up(&dk), up(&dv), up(&cot));
            let (qc, kc, vc, dqc, dkc, dvc, cc) = (up(&call(&q)), up(&call(&k)), up(&call(&v)), up(&call(&dq)), up(&call(&dk)), up(&call(&dv)), up(&call(&cot)));
            for causal in [false, true] {
                let full = d.download(&forward(&d, (&qw, &kw, &vw), 3, 0.25, causal, Arithmetic::F64).expect("forward")).expect("download");
                close("forward", &forward_segments(&d, (&qc, &kc, &vc), &segments, 0.25, causal, Arithmetic::F64).expect("segments"), &call(&full));
                let (gq, gk, gv) = backward(&d, (&qw, &kw, &vw), &cw, 3, 0.25, causal, (Arithmetic::F64, Arithmetic::F64)).expect("backward");
                // The second sequence's rows before `first` feed the segment through the first's.
                let folded = |g: &Tensor| {
                    let mut g = d.download(g).expect("download");
                    for r in 0..first {
                        let row = g.row(length + r).to_owned();
                        let mut target = g.row_mut(r);
                        target += &row;
                    }
                    call(&g)
                };
                let (sq, sk, sv) = backward_segments(&d, (&qc, &kc, &vc), &cc, &segments, 0.25, causal, (Arithmetic::F64, Arithmetic::F64)).expect("segments");
                close("query cotangent", &sq, &call(&d.download(&gq).expect("download")));
                close("key cotangent", &sk, &folded(&gk));
                close("value cotangent", &sv, &folded(&gv));
                let t = d.download(&tangent(&d, (&qw, &kw, &vw), (Some(&dqw), Some(&dkw), Some(&dvw)), 3, 0.25, causal, (Arithmetic::F64, Arithmetic::F64)).expect("tangent")).expect("download");
                close("tangent", &tangent_segments(&d, (&qc, &kc, &vc), (Some(&dqc), Some(&dkc), Some(&dvc)), &segments, 0.25, causal, (Arithmetic::F64, Arithmetic::F64)).expect("segments"), &call(&t));
            }
        }
    }
    #[test]
    fn resident_tiles_match_host_attention_across_tile_and_sequence_boundaries() {
        use crate::operator_program::{FamilyInputs, SequenceLayout};
        let length = 263;
        let rows = 2 * length;
        let data = |width, salt| Array2::from_shape_fn((rows, width), |(r, c)| ((r * 31 + c * 7 + salt) as f64 * 0.73).sin());
        let (q, k, v, dq, dk, dv, cot) = (data(8, 1), data(8, 2), data(5, 3), data(8, 4), data(8, 5), data(5, 6), data(5, 7));
        let family = FamilyInputs { rows, slots: vec![], layout: Some(SequenceLayout { sequence: (0..rows).map(|r| (r / length) as u32).collect(), position: (0..rows).map(|r| (r % length) as u32).collect() }) };
        for d in crate::device_program_tests::devices() {
            let up = |x: &Array2<f64>| d.upload(x.view()).expect("upload");
            let (qd, kd, vd, dqd, dkd, dvd, cd) = (up(&q), up(&k), up(&v), up(&dq), up(&dk), up(&dv), up(&cot));
            let close = |actual: &Tensor, expected: &Array2<f64>| {
                let actual = d.download(actual).expect("download");
                let error = (&actual - expected).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                assert!(error < 1e-10, "{}: max error {error}", d.name());
            };
            for causal in [false, true] {
                close(&forward(&d, (&qd, &kd, &vd), 2, 0.25, causal, Arithmetic::F64).expect("forward"), &crate::tiled_attention::forward(&family, (&q, &k, &v), 0.25, None, causal).expect("reference"));
                let actual = backward(&d, (&qd, &kd, &vd), &cd, 2, 0.25, causal, (Arithmetic::F64, Arithmetic::F64)).expect("backward");
                let expected = crate::tiled_attention::backward(&family, (&q, &k, &v), &cot, 0.25, None, causal).expect("reference");
                close(&actual.0, &expected.0); close(&actual.1, &expected.1); close(&actual.2, &expected.2);
                let actual = tangent(&d, (&qd, &kd, &vd), (Some(&dqd), Some(&dkd), Some(&dvd)), 2, 0.25, causal, (Arithmetic::F64, Arithmetic::F64)).expect("tangent");
                let expected = crate::tiled_attention::tangent(&family, (&q, &k, &v), (Some(&dq), Some(&dk), Some(&dv)), 0.25, None, causal).expect("reference");
                close(&actual, &expected);
            }
        }
    }
}

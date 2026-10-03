//! Tiled resident attention using the existing tensor backend. The fast full-matrix route is
//! retained for small batches; this route bounds sequence-square scratch in larger workloads.
use gam_gpu::{gpu_error::GpuError, tensor::{Arithmetic, Device, Op, Tensor}};
const TILE: usize = 256;
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

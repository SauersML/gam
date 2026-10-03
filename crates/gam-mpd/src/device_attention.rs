//! Tiled resident attention using the existing tensor backend. The fast full-matrix route is
//! retained for small batches; this route bounds sequence-square scratch in larger workloads.
use gam_gpu::{gpu_error::GpuError, tensor::{Arithmetic, Device, Op, Tensor}};
const TILE: usize = 256;
type Triple = (Tensor, Tensor, Tensor);
type Values<'a> = (&'a Tensor, &'a Tensor, &'a Tensor);

fn probabilities(d: &Device, q: &Tensor, k: &Tensor, start: usize, scale: f64, causal: bool) -> Result<Tensor, GpuError> {
    let mut p = d.zeros(q.rows(), k.rows())?;
    d.gemm(&mut p, scale, q, Op::N, k, Op::T, 0.0, Arithmetic::F64)?;
    d.softmax_rows_offset(&mut p, causal, start)?;
    Ok(p)
}

pub(crate) fn forward(d: &Device, (q, k, v): Values<'_>, blocks: usize, scale: f64, causal: bool) -> Result<Tensor, GpuError> {
    let length = q.rows() / blocks;
    let mut out = d.zeros(q.rows(), v.cols())?;
    for block in 0..blocks {
        let base = block * length;
        let ks = d.rows_of(k, base, length)?;
        let vs = d.rows_of(v, base, length)?;
        for start in (0..length).step_by(TILE) {
            let n = TILE.min(length - start);
            let qb = d.rows_of(q, base + start, n)?;
            let p = probabilities(d, &qb, &ks, start, scale, causal)?;
            let mut tile = d.zeros(n, v.cols())?;
            d.gemm(&mut tile, 1.0, &p, Op::N, &vs, Op::N, 0.0, Arithmetic::F64)?;
            d.set_rows(&mut out, base + start, &tile)?;
        }
    }
    Ok(out)
}

pub(crate) fn backward(d: &Device, (q, k, v): Values<'_>, cot: &Tensor, blocks: usize, scale: f64, causal: bool, arithmetic: Arithmetic) -> Result<Triple, GpuError> {
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
            let p = probabilities(d, &qb, &ks, start, scale, causal)?;
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

pub(crate) fn tangent(d: &Device, (q, k, v): Values<'_>, (dq, dk, dv): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>), blocks: usize, scale: f64, causal: bool, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
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
            let p = probabilities(d, &qb, &ks, start, scale, causal)?;
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

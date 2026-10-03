//! Bounded-scratch F64 attention for ordinary execution and fitting. Query tiles use matrix
//! products against one sequence's keys. The banded evaluator keeps its separate reference path.
use super::operator_program::{FamilyInputs, ProgramError, Rotary, SequenceLayout};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
use ndarray::{Array2, ArrayView2, Axis, s};
use std::collections::BTreeMap;

const TILE: usize = 64;
type Triple = (Array2<f64>, Array2<f64>, Array2<f64>);
type Values<'a> = (&'a Array2<f64>, &'a Array2<f64>, &'a Array2<f64>);

fn layout(inputs: &FamilyInputs) -> Result<&SequenceLayout, ProgramError> {
    let layout = inputs.layout.as_ref().ok_or_else(|| ProgramError::Input("attention needs a layout".to_string()))?;
    if layout.sequence.len() != inputs.rows || layout.position.len() != inputs.rows {
        return Err(ProgramError::Input("attention layout length mismatch".to_string()));
    }
    Ok(layout)
}

fn groups(layout: &SequenceLayout) -> BTreeMap<u32, Vec<usize>> {
    let mut out = BTreeMap::<u32, Vec<usize>>::new();
    for (row, sequence) in layout.sequence.iter().enumerate() { out.entry(*sequence).or_default().push(row); }
    out
}

fn rotate(x: &Array2<f64>, rotary: Option<Rotary>, positions: &[u32], inverse: bool) -> Array2<f64> {
    let mut out = x.clone();
    if let Some(rotary) = rotary {
        let pairs = rotary.pairs();
        for (row, position) in positions.iter().enumerate() {
            for (plane, &(a, b)) in pairs.iter().enumerate() {
                let (c, s) = rotary.turn(plane, *position);
                let s = if inverse { -s } else { s };
                let (x, y) = (out[[row, a]], out[[row, b]]);
                out[[row, a]] = c * x - s * y;
                out[[row, b]] = s * x + c * y;
            }
        }
    }
    out
}

fn probabilities(q: ArrayView2<'_, f64>, k: &Array2<f64>, positions: &[u32], start: usize, scale: f64, causal: bool) -> Array2<f64> {
    let mut p = fast_abt(&q, k);
    for (r, mut row) in p.outer_iter_mut().enumerate() {
        let mut max = f64::NEG_INFINITY;
        for (c, v) in row.iter_mut().enumerate() {
            *v = if causal && positions[c] > positions[start + r] { f64::NEG_INFINITY } else { *v * scale };
            max = max.max(*v);
        }
        let mut total = 0.0;
        for v in &mut row { *v = (*v - max).exp(); total += *v; }
        row /= total;
    }
    p
}

fn softmax_derivative(p: &Array2<f64>, d: &mut Array2<f64>, scale: f64) {
    for (pr, mut dr) in p.outer_iter().zip(d.outer_iter_mut()) {
        let mean = pr.dot(&dr);
        for (a, v) in pr.iter().zip(dr.iter_mut()) { *v = scale * *a * (*v - mean); }
    }
}

fn scatter(out: &mut Array2<f64>, members: &[usize], values: &Array2<f64>) {
    for (local, row) in members.iter().enumerate() { out.row_mut(*row).assign(&values.row(local)); }
}

pub(crate) fn forward(inputs: &FamilyInputs, (query, key, value): Values<'_>, scale: f64, rotary: Option<Rotary>, causal: bool) -> Result<Array2<f64>, ProgramError> {
    let layout = layout(inputs)?;
    let q = rotate(query, rotary, &layout.position, false);
    let k = rotate(key, rotary, &layout.position, false);
    let mut out = Array2::zeros((inputs.rows, value.ncols()));
    for members in groups(layout).values() {
        let (q, k, v) = (q.select(Axis(0), members), k.select(Axis(0), members), value.select(Axis(0), members));
        let positions: Vec<_> = members.iter().map(|r| layout.position[*r]).collect();
        for start in (0..members.len()).step_by(TILE) {
            let end = (start + TILE).min(members.len());
            let p = probabilities(q.slice(s![start..end, ..]), &k, &positions, start, scale, causal);
            scatter(&mut out, &members[start..end], &fast_ab(&p, &v));
        }
    }
    Ok(out)
}

pub(crate) fn backward(inputs: &FamilyInputs, (query, key, value): Values<'_>, cotangent: &Array2<f64>, scale: f64, rotary: Option<Rotary>, causal: bool) -> Result<Triple, ProgramError> {
    let layout = layout(inputs)?;
    let q = rotate(query, rotary, &layout.position, false);
    let k = rotate(key, rotary, &layout.position, false);
    let (mut gq, mut gk, mut gv) = (Array2::zeros(q.dim()), Array2::zeros(k.dim()), Array2::zeros(value.dim()));
    for members in groups(layout).values() {
        let (q, k, v, cot) = (q.select(Axis(0), members), k.select(Axis(0), members), value.select(Axis(0), members), cotangent.select(Axis(0), members));
        let positions: Vec<_> = members.iter().map(|r| layout.position[*r]).collect();
        let (mut key_grad, mut value_grad) = (Array2::zeros(k.dim()), Array2::zeros(v.dim()));
        for start in (0..members.len()).step_by(TILE) {
            let end = (start + TILE).min(members.len());
            let qb = q.slice(s![start..end, ..]);
            let cb = cot.slice(s![start..end, ..]);
            let p = probabilities(qb, &k, &positions, start, scale, causal);
            let mut ds = fast_abt(&cb, &v);
            softmax_derivative(&p, &mut ds, scale);
            scatter(&mut gq, &members[start..end], &fast_ab(&ds, &k));
            key_grad += &fast_atb(&ds, &qb);
            value_grad += &fast_atb(&p, &cb);
        }
        scatter(&mut gk, members, &key_grad);
        scatter(&mut gv, members, &value_grad);
    }
    Ok((rotate(&gq, rotary, &layout.position, true), rotate(&gk, rotary, &layout.position, true), gv))
}

pub(crate) fn tangent(inputs: &FamilyInputs, (query, key, value): Values<'_>, (dq, dk, dv): (Option<&Array2<f64>>, Option<&Array2<f64>>, Option<&Array2<f64>>), scale: f64, rotary: Option<Rotary>, causal: bool) -> Result<Array2<f64>, ProgramError> {
    let layout = layout(inputs)?;
    let q = rotate(query, rotary, &layout.position, false);
    let k = rotate(key, rotary, &layout.position, false);
    let dq = dq.map(|v| rotate(v, rotary, &layout.position, false));
    let dk = dk.map(|v| rotate(v, rotary, &layout.position, false));
    let mut out = Array2::zeros((inputs.rows, value.ncols()));
    for members in groups(layout).values() {
        let (q, k, v) = (q.select(Axis(0), members), k.select(Axis(0), members), value.select(Axis(0), members));
        let (dq, dk, dv) = (dq.as_ref().map(|v| v.select(Axis(0), members)), dk.as_ref().map(|v| v.select(Axis(0), members)), dv.map(|v| v.select(Axis(0), members)));
        let positions: Vec<_> = members.iter().map(|r| layout.position[*r]).collect();
        for start in (0..members.len()).step_by(TILE) {
            let end = (start + TILE).min(members.len());
            let qb = q.slice(s![start..end, ..]);
            let p = probabilities(qb, &k, &positions, start, scale, causal);
            let mut ds = Array2::zeros(p.dim());
            if let Some(dq) = &dq { ds += &fast_abt(&dq.slice(s![start..end, ..]), &k); }
            if let Some(dk) = &dk { ds += &fast_abt(&qb, dk); }
            softmax_derivative(&p, &mut ds, scale);
            let mut tile = fast_ab(&ds, &v);
            if let Some(dv) = &dv { tile += &fast_ab(&p, dv); }
            scatter(&mut out, &members[start..end], &tile);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn data(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| ((i * 31 + j * 7 + salt) as f64 * 0.73).sin())
    }
    fn reference(inputs: &FamilyInputs, (query, key, value): Values<'_>, scale: f64, rotary: Option<Rotary>, causal: bool) -> Array2<f64> {
        let layout = inputs.layout.as_ref().expect("layout");
        let rotate_reference = |x: &Array2<f64>| {
            let mut out = x.clone();
            if let Some(rotary) = rotary {
                for (r, mut row) in out.outer_iter_mut().enumerate() {
                    let mut values = row.to_vec();
                    rotary.rotate(&mut values, None, layout.position[r]);
                    row.assign(&ndarray::ArrayView1::from(&values));
                }
            }
            out
        };
        let (q, k) = (rotate_reference(query), rotate_reference(key));
        let mut out = Array2::zeros((inputs.rows, value.ncols()));
        for members in groups(layout).values() {
            for &r in members {
                let keys: Vec<_> = members.iter().copied().filter(|s| !causal || layout.position[*s] <= layout.position[r]).collect();
                let scores: Vec<f64> = keys.iter().map(|s| scale * q.row(r).dot(&k.row(*s))).collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = e.iter().sum();
                for (&s, e) in keys.iter().zip(e) { out.row_mut(r).scaled_add(e / total, &value.row(s)); }
            }
        }
        out
    }
    fn close(a: &Array2<f64>, b: &Array2<f64>, tolerance: f64) {
        assert_eq!(a.dim(), b.dim());
        let max = (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(max < tolerance, "max difference {max}");
    }
    #[test]
    fn tiled_attention_preserves_masks_rotations_and_both_derivatives() {
        let rows = 205;
        let inputs = FamilyInputs { rows, slots: vec![], layout: Some(SequenceLayout {
            sequence: (0..rows).map(|r| (r % 2) as u32).collect(),
            position: (0..rows).map(|r| ((r * 7) % 53) as u32).collect(),
        }) };
        let (q, k, v) = (data(rows, 8, 1), data(rows, 8, 2), data(rows, 5, 3));
        let (dq, dk, dv, cot) = (data(rows, 8, 4), data(rows, 8, 5), data(rows, 5, 6), data(rows, 5, 7));
        for causal in [false, true] {
            for rotary in [None, Some(Rotary { base: 10000, dims: 6, half_split: false }), Some(Rotary { base: 10000, dims: 6, half_split: true })] {
                let actual = forward(&inputs, (&q, &k, &v), 0.25, rotary, causal).expect("forward");
                close(&actual, &reference(&inputs, (&q, &k, &v), 0.25, rotary, causal), 1e-12);
                let jvp = tangent(&inputs, (&q, &k, &v), (Some(&dq), Some(&dk), Some(&dv)), 0.25, rotary, causal).expect("tangent");
                let eps = 1e-5;
                let plus = forward(&inputs, (&(&q + &(&dq * eps)), &(&k + &(&dk * eps)), &(&v + &(&dv * eps))), 0.25, rotary, causal).expect("plus");
                let minus = forward(&inputs, (&(&q - &(&dq * eps)), &(&k - &(&dk * eps)), &(&v - &(&dv * eps))), 0.25, rotary, causal).expect("minus");
                close(&jvp, &((plus - minus) / (2.0 * eps)), 1e-8);
                let (gq, gk, gv) = backward(&inputs, (&q, &k, &v), &cot, 0.25, rotary, causal).expect("backward");
                let left = (&cot * &jvp).sum();
                let right = (&gq * &dq).sum() + (&gk * &dk).sum() + (&gv * &dv).sum();
                assert!((left - right).abs() < 1e-9, "adjoint {left} vs {right}");
                let mut changed = v.clone();
                for r in (1..rows).step_by(2) { changed.row_mut(r).fill(100.0); }
                let separate = forward(&inputs, (&q, &k, &changed), 0.25, rotary, causal).expect("independent sequences");
                for r in (0..rows).step_by(2) { assert_eq!(separate.row(r), actual.row(r)); }
            }
        }
    }
    #[test]
    #[ignore = "manual attention timing; no timing assertion"]
    fn benchmark_training_attention() {
        use std::{hint::black_box, time::Instant};
        for rows in [256, 1024] {
            let inputs = FamilyInputs { rows, slots: vec![], layout: Some(SequenceLayout { sequence: vec![0; rows], position: (0..rows as u32).collect() }) };
            let (q, k, v) = (data(rows, 64, 1), data(rows, 64, 2), data(rows, 64, 3));
            let mut old = Vec::new(); let mut new = Vec::new();
            for _ in 0..3 {
                let started = Instant::now();
                let expected = black_box(reference(&inputs, (&q, &k, &v), 0.125, None, true));
                old.push(started.elapsed().as_secs_f64());
                let started = Instant::now();
                let actual = black_box(forward(&inputs, (&q, &k, &v), 0.125, None, true).expect("forward"));
                new.push(started.elapsed().as_secs_f64());
                close(&actual, &expected, 1e-12);
            }
            old.sort_by(f64::total_cmp); new.sort_by(f64::total_cmp);
            eprintln!("attention rows={rows} width=64 scalar_seconds={} tiled_seconds={} ratio={}", old[1], new[1], old[1] / new[1]);
        }
    }
}

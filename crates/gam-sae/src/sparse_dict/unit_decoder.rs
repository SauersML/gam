//! Descent on unit decoder rows with the sparse codes held fixed.
//!
//! Normalizing an unconstrained MOD solve is not a constrained minimizer when
//! atoms co-fire. On unit rows the diagonal quadratic and decoder ridge are
//! constant. Row a therefore minimizes its conditional loss by aligning with
//! b_a - sum_{b != a} A_ab d_b. A deterministic Gauss–Seidel sweep decreases
//! that same objective, using only the occupied co-firing graph.

use super::update::DecoderNormalEq;
use ndarray::Array2;

pub(super) fn refine(
    decoder: &mut Array2<f32>,
    previous: &Array2<f32>,
    eq: &DecoderNormalEq,
    admitted: &[bool],
) -> Result<(), String> {
    let k = decoder.nrows();
    let p = decoder.ncols();
    let mut neighbors = vec![Vec::new(); k];
    for (&(a, b), &weight) in &eq.off {
        neighbors[a as usize].push((b as usize, weight));
        neighbors[b as usize].push((a as usize, weight));
    }
    for row in &mut neighbors {
        row.sort_unstable_by_key(|&(atom, _)| atom);
    }
    // The normalized unconstrained solution is only a proposal. Keep a flat
    // row's existing direction and reject a proposal that increases the fixed-
    // code objective before taking the constrained coordinate sweep.
    for atom in 0..k {
        if !admitted[atom] || decoder.row(atom).iter().all(|&v| v == 0.0) {
            decoder.row_mut(atom).assign(&previous.row(atom));
        }
    }
    let variable_loss = |directions: &Array2<f32>| -> f64 {
        let mut value = 0.0;
        for atom in 0..k {
            for feature in 0..p {
                value -= 2.0 * eq.b[[atom, feature]] * directions[[atom, feature]] as f64;
            }
            for &(other, weight) in &neighbors[atom] {
                if other > atom {
                    value += 2.0
                        * weight
                        * directions
                            .row(atom)
                            .iter()
                            .zip(directions.row(other).iter())
                            .map(|(&a, &b)| a as f64 * b as f64)
                            .sum::<f64>();
                }
            }
        }
        value
    };
    if variable_loss(decoder) > variable_loss(previous) {
        decoder.assign(previous);
    }
    let mut residual = vec![0.0_f64; p];
    for atom in 0..k {
        if !admitted[atom] {
            continue;
        }
        for feature in 0..p {
            residual[feature] = eq.b[[atom, feature]];
        }
        for &(other, weight) in &neighbors[atom] {
            for feature in 0..p {
                residual[feature] -= weight * decoder[[other, feature]] as f64;
            }
        }
        let norm = residual
            .iter()
            .map(|&value| value * value)
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return Err(format!(
                "unit decoder row {atom} has a non-finite conditional residual"
            ));
        }
        if norm == 0.0 {
            continue;
        }
        let old_alignment: f64 = residual
            .iter()
            .zip(decoder.row(atom).iter())
            .map(|(&r, &d)| r * d as f64)
            .sum();
        let new_alignment: f64 = residual
            .iter()
            .map(|&r| r * ((r / norm) as f32) as f64)
            .sum();
        if new_alignment > old_alignment {
            for feature in 0..p {
                decoder[[atom, feature]] = (residual[feature] / norm) as f32;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn normalized_mod_fixed_point_is_not_a_unit_constrained_stationary_point() {
        // A is SPD, and B=A*[[2,0],[0,1]]. Normalized MOD returns identity
        // forever, although its second row has nonzero tangent gradient.
        let mut eq = DecoderNormalEq::zeros(2, 2);
        eq.diag = vec![2.0, 2.0];
        eq.off.insert((0, 1), 1.0);
        eq.b = array![[4.0, 1.0], [2.0, 2.0]];
        let previous = Array2::eye(2);
        let mut decoder = previous.clone();
        refine(&mut decoder, &previous, &eq, &[true, true]).unwrap();
        assert_eq!(decoder.row(0), previous.row(0));
        let second = [1.0_f64 / 5.0_f64.sqrt(), 2.0 / 5.0_f64.sqrt()];
        for feature in 0..2 {
            assert!((decoder[[1, feature]] as f64 - second[feature]).abs() < 1e-7);
        }
        let loss = |d: &Array2<f32>| {
            4.0 + 2.0 * d.row(0).dot(&d.row(1)) as f64
                - 2.0
                    * eq.b
                        .iter()
                        .zip(d.iter())
                        .map(|(&b, &d)| b * d as f64)
                        .sum::<f64>()
        };
        assert!(loss(&decoder) < loss(&previous) - 0.4);
    }

    #[test]
    fn deferred_and_flat_rows_keep_their_existing_direction() {
        let mut eq = DecoderNormalEq::zeros(2, 2);
        eq.b = array![[0.0, 0.0], [3.0, -1.0]];
        let previous = Array2::eye(2);
        let mut decoder = Array2::zeros((2, 2));
        refine(&mut decoder, &previous, &eq, &[true, false]).unwrap();
        assert_eq!(decoder, previous);
    }
}

//! #2822 — the per-state half of the softmax residual-curvature HVP: every row's
//! residual-probe contractions of its second-order channels.
//!
//! [`SaeRowJetContraction::Bilinear`](super::SaeRowJetContraction) reads a row's channels
//! only through
//!
//! ```text
//! G[a][b] = ⟨probe, second(a, b)⟩   (q × q),     C[a][c] = ⟨probe, beta_deriv(a, c)⟩   (q × n_β),
//! ```
//!
//! and both depend on the state alone: the direction `(v_t, v_β)` enters only the final
//! `t = G v_t + C v_β`, `β = Cᵀ v_t`. The CPU tile nonetheless re-executed the row
//! program, into a freshly zeroed packed channel buffer, for every row of every HVP. On
//! the zoo rank-charge row at 4 threads that was 10 of 11 stack samples after the fold
//! fix (sw2f 1249621, #2822). [`prepare_bilinear_contractions`] runs the row program
//! once per prepared state, and [`SaeRowJetBilinearContractions::apply`] contracts a
//! direction against the kept words. `G` and `C` are the same `dot` of the same channel
//! words the tile takes, and `t` and `β` add them in the tile's order from the same
//! `+0.0`, so an apply is bit-identical to the tile it replaces.

use super::{
    InputSource, SaeRowJetContractedTile, SaeSoftmaxRowJetInput, checked_product, finite_or_err,
    validate_tile,
};
use crate::row_jet_program::execute_softmax_row_program;
use gam_runtime::resource::{Governed, MemoryGovernor};

/// Every row's `G` then `C`, row-major, charged to the memory governor.
pub(crate) struct SaeRowJetBilinearContractions {
    n_rows: usize,
    q: usize,
    n_beta: usize,
    rows: Governed<Vec<f64>>,
}

/// The CPU tile's `dot`, so a kept contraction is the word the tile computes.
fn contraction_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Run each row's program once and keep `G` and `C` against its probe row (`probe` is
/// row-major `n_rows × p`). `None` when the governor declines the `n_rows·q·(q + n_β)`
/// words; the caller then keeps contracting through the tile.
pub(crate) fn prepare_bilinear_contractions(
    rows: &[SaeSoftmaxRowJetInput],
    inv_tau: f64,
    probe: &[f64],
) -> Result<Option<SaeRowJetBilinearContractions>, String> {
    if !inv_tau.is_finite() || inv_tau <= 0.0 {
        return Err(format!(
            "SAE row-jet inverse temperature must be finite and positive; got {inv_tau}"
        ));
    }
    let shape = validate_tile(rows)?;
    let (q, p, n_beta) = (shape.1, shape.2, shape.3);
    let n = rows.len();
    let probe_len = checked_product(&[n, p])?;
    if probe.len() != probe_len {
        return Err(format!(
            "SAE row-jet contraction probe length {} != expected {probe_len}",
            probe.len()
        ));
    }
    finite_or_err("probe", probe)?;
    let per_row = checked_product(&[q, q])?
        .checked_add(checked_product(&[q, n_beta])?)
        .ok_or_else(|| "SAE row-jet bilinear contraction row length overflows".to_string())?;
    let charge = match MemoryGovernor::global().try_reserve_dense_f64(
        n,
        per_row,
        "SAE row-jet bilinear contractions",
    ) {
        Ok(charge) => charge,
        Err(refusal) => {
            log::debug!("[SAE row-jet] {refusal}; contracting through the tile per apply");
            return Ok(None);
        }
    };
    let mut values = vec![0.0_f64; checked_product(&[n, per_row])?];
    for (row, input) in rows.iter().enumerate() {
        let source = InputSource::new(input);
        let scheduled = execute_softmax_row_program(&source, inv_tau, input.sqrt_row_weight);
        source.finish()?;
        let probe_row = &probe[row * p..(row + 1) * p];
        let (second, mixed) = values[row * per_row..(row + 1) * per_row].split_at_mut(q * q);
        for a in 0..q {
            for b in 0..q {
                second[a * q + b] = contraction_dot(probe_row, scheduled.second(a, b));
            }
            for border in 0..n_beta {
                mixed[a * n_beta + border] =
                    contraction_dot(probe_row, scheduled.beta_deriv(a, border));
            }
        }
    }
    Ok(Some(SaeRowJetBilinearContractions {
        n_rows: n,
        q,
        n_beta,
        rows: charge.bind(values),
    }))
}

impl SaeRowJetBilinearContractions {
    /// `t[r][a] = Σ_b G_r[a][b] v_t[r][b] + Σ_c C_r[a][c] v_β[r][c]` and
    /// `β[r][c] = Σ_a C_r[a][c] v_t[r][a]`, for row-major `v_t` (`n_rows × q`) and `v_β`
    /// (`n_rows × n_β`), under the tile's length and finiteness checks.
    pub(crate) fn apply(
        &self,
        v_t: &[f64],
        v_beta: &[f64],
    ) -> Result<SaeRowJetContractedTile, String> {
        let (n, q, n_beta) = (self.n_rows, self.q, self.n_beta);
        for (label, got, want) in [
            ("v_t", v_t.len(), checked_product(&[n, q])?),
            ("v_beta", v_beta.len(), checked_product(&[n, n_beta])?),
        ] {
            if got != want {
                return Err(format!(
                    "SAE row-jet contraction {label} length {got} != expected {want}"
                ));
            }
        }
        finite_or_err("v_t", v_t)?;
        finite_or_err("v_beta", v_beta)?;
        let per_row = q * q + q * n_beta;
        let mut t = vec![0.0_f64; n * q];
        let mut beta = vec![0.0_f64; n * n_beta];
        for row in 0..n {
            let (second, mixed) = self.rows[row * per_row..(row + 1) * per_row].split_at(q * q);
            let v_t_row = &v_t[row * q..(row + 1) * q];
            let v_beta_row = &v_beta[row * n_beta..(row + 1) * n_beta];
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    acc += second[a * q + b] * v_t_row[b];
                }
                for border in 0..n_beta {
                    acc += mixed[a * n_beta + border] * v_beta_row[border];
                }
                t[row * q + a] = acc;
            }
            for border in 0..n_beta {
                let mut acc = 0.0_f64;
                for a in 0..q {
                    acc += mixed[a * n_beta + border] * v_t_row[a];
                }
                beta[row * n_beta + border] = acc;
            }
        }
        Ok(SaeRowJetContractedTile {
            n_rows: n,
            q,
            n_beta,
            t,
            beta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::prepare_bilinear_contractions;
    use crate::gpu_kernels::sae_rowjet::{
        SaeRowJetContraction, SaeRowJetPath, SaeRowJetPrimary, SaeSoftmaxRowJetInput,
        execute_softmax_row_jet_tile_contracted,
    };

    /// Softmax rows over three atoms with two logits and four coordinate primaries, every
    /// channel family populated, and the third atom inactive on odd rows so some channels
    /// are exact zeros.
    fn rows(n: usize) -> Vec<SaeSoftmaxRowJetInput> {
        let (k, p) = (3usize, 2usize);
        let primaries = vec![
            SaeRowJetPrimary::Logit { atom: 0 },
            SaeRowJetPrimary::Logit { atom: 1 },
            SaeRowJetPrimary::Coordinate { atom: 0, axis: 0 },
            SaeRowJetPrimary::Coordinate { atom: 1, axis: 0 },
            SaeRowJetPrimary::Coordinate { atom: 1, axis: 1 },
            SaeRowJetPrimary::Coordinate { atom: 2, axis: 0 },
        ];
        let q = primaries.len();
        (0..n)
            .map(|row| {
                let logits: Vec<f64> = (0..k)
                    .map(|atom| 0.4 * ((row * 17 + atom * 11 + 1) as f64 * 0.07).sin())
                    .collect();
                let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|value| (value - shift).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let mut decoded_first = vec![0.0; q * p];
                for slot in 2..q {
                    for c in 0..p {
                        decoded_first[slot * p + c] =
                            ((row * 19 + slot * 5 + c + 2) as f64 * 0.04).sin();
                    }
                }
                let mut decoded_second = vec![0.0; q * q * p];
                for a in 2..q {
                    for b in 2..q {
                        let same_atom = match (primaries[a], primaries[b]) {
                            (
                                SaeRowJetPrimary::Coordinate { atom: left, .. },
                                SaeRowJetPrimary::Coordinate { atom: right, .. },
                            ) => left == right,
                            _ => false,
                        };
                        if same_atom {
                            for c in 0..p {
                                decoded_second[(a * q + b) * p + c] =
                                    ((row * 23 + a * 7 + b * 3 + c + 1) as f64 * 0.03).cos();
                            }
                        }
                    }
                }
                let n_beta = 3;
                let mut beta_basis_first = vec![0.0; q * n_beta];
                beta_basis_first[2 * n_beta] = 0.2;
                beta_basis_first[3 * n_beta + 1] = -0.4;
                beta_basis_first[4 * n_beta + 1] = 0.7;
                beta_basis_first[5 * n_beta + 2] = -0.1;
                SaeSoftmaxRowJetInput {
                    n_atoms: k,
                    out_dim: p,
                    coordinate_slots: SaeSoftmaxRowJetInput::coordinate_slots_for(&primaries),
                    primaries: primaries.clone(),
                    gate_values: exps.iter().map(|value| value / sum).collect(),
                    active_atoms: vec![true, true, row % 2 == 0],
                    sqrt_row_weight: (1.0 + row as f64 * 0.1).sqrt(),
                    decoded: (0..k * p)
                        .map(|index| ((row * 13 + index * 7 + 3) as f64 * 0.09).cos())
                        .collect(),
                    decoded_first,
                    decoded_second,
                    beta_atoms: vec![0, 1, 2].into(),
                    beta_basis_values: vec![0.8, -0.3, 0.5],
                    beta_basis_first,
                    beta_outputs: vec![1.0, 0.2, -0.5, 0.8, 0.3, -0.7].into(),
                }
            })
            .collect()
    }

    fn words(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Every apply of the kept contractions writes the CPU tile's words, `t` and `β`, for
    /// several directions against one state. A second probe gives other words, so the
    /// equality is not two zero outputs agreeing.
    #[test]
    fn kept_bilinear_contractions_apply_the_cpu_tiles_words_2822() {
        let (n, q, n_beta, p) = (16usize, 6usize, 3usize, 2usize);
        let inputs = rows(n);
        let inv_tau = 1.0 / 0.7;
        let probe: Vec<f64> = (0..n * p)
            .map(|index| ((index * 29 + 5) as f64 * 0.013).sin())
            .collect();
        let kept = prepare_bilinear_contractions(&inputs, inv_tau, &probe)
            .expect("the fixture's rows run their program")
            .expect("a 16-row fixture fits the governor");
        let mut checked = 0usize;
        for direction in 0..4 {
            let v_t: Vec<f64> = (0..n * q)
                .map(|index| match direction {
                    0 => 1.0,
                    1 => ((index * 7 + direction) as f64 * 0.21).cos(),
                    2 => if index % 3 == 0 { -2.5 } else { 0.0 },
                    _ => ((index * 11 + 3) as f64 * 0.17).sin() * 1.0e3,
                })
                .collect();
            let v_beta: Vec<f64> = (0..n * n_beta)
                .map(|index| ((index * 5 + direction) as f64 * 0.31).sin())
                .collect();
            let tile = execute_softmax_row_jet_tile_contracted(
                &inputs,
                inv_tau,
                SaeRowJetPath::Cpu,
                SaeRowJetContraction::Bilinear {
                    probe: &probe,
                    v_t: &v_t,
                    v_beta: &v_beta,
                },
            )
            .expect("CPU contracted tile");
            let applied = kept.apply(&v_t, &v_beta).expect("kept contraction apply");
            assert_eq!(
                (applied.n_rows, applied.q, applied.n_beta),
                (tile.n_rows, tile.q, tile.n_beta),
                "direction {direction}: the kept contraction's shape differs from the tile's"
            );
            assert_eq!(
                (words(&applied.t), words(&applied.beta)),
                (words(&tile.t), words(&tile.beta)),
                "direction {direction}: the kept contraction's words differ from the CPU tile's"
            );
            checked += tile.t.iter().filter(|value| **value != 0.0).count();
        }
        assert!(checked > 0, "every compared t word was zero, so the equality checked nothing");

        let other_probe: Vec<f64> = probe.iter().map(|value| value * 1.5 + 0.25).collect();
        let other = prepare_bilinear_contractions(&inputs, inv_tau, &other_probe)
            .expect("the fixture's rows run their program")
            .expect("a 16-row fixture fits the governor");
        let ones_t = vec![1.0; n * q];
        let ones_beta = vec![1.0; n * n_beta];
        assert_ne!(
            words(&other.apply(&ones_t, &ones_beta).expect("apply").t),
            words(&kept.apply(&ones_t, &ones_beta).expect("apply").t),
            "control failed: a different probe gave the same words, so word equality cannot \
             tell two probes apart on this fixture"
        );
    }

    /// The kept contraction refuses what the tile refuses: a non-finite or wrongly sized
    /// direction, and a non-finite probe at preparation.
    #[test]
    fn kept_bilinear_contractions_refuse_what_the_tile_refuses_2822() {
        let (n, q, n_beta, p) = (4usize, 6usize, 3usize, 2usize);
        let inputs = rows(n);
        let probe = vec![0.5; n * p];
        let kept = prepare_bilinear_contractions(&inputs, 1.0, &probe)
            .expect("the fixture's rows run their program")
            .expect("a 4-row fixture fits the governor");
        let mut v_t = vec![0.1; n * q];
        let v_beta = vec![0.2; n * n_beta];
        assert!(kept.apply(&v_t, &v_beta).is_ok(), "a finite direction must apply");
        v_t[3] = f64::NAN;
        assert!(kept.apply(&v_t, &v_beta).is_err(), "a NaN direction entry must be refused");
        assert!(
            kept.apply(&v_t[..n * q - 1], &v_beta).is_err(),
            "a short direction must be refused"
        );
        let mut bad_probe = probe.clone();
        bad_probe[1] = f64::INFINITY;
        assert!(
            prepare_bilinear_contractions(&inputs, 1.0, &bad_probe).is_err(),
            "a non-finite probe must be refused at preparation"
        );
        assert!(
            prepare_bilinear_contractions(&inputs, 0.0, &probe).is_err(),
            "a zero inverse temperature must be refused"
        );
    }
}

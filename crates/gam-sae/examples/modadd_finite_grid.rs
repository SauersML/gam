//! Exhaustive FANOVA of a modular-addition model's logits on `Z_p²` (#2946 R7, finite grid).
//!
//! Input: the model's logits at `=` on every `(a, b)`, row-major (`row = a·p + b`), as little-endian
//! f64 `p² × p` — e.g. `bench/mpd_opfirst_modadd_program_2951.py`'s `load`/`model_forward` output written
//! with `logits.astype("<f8").tofile(path)`. Reports, under the uniform law on `Z_p²` with centred
//! logits: the interaction share and rectangle bound in `(a, b)`, the same in `(s, d) = (a+b, a−b)`,
//! and the argmax accuracy of the `s`-only law `E[logits | s]` on all `p²` inputs.
//!
//! Usage: `modadd_finite_grid <logits.f64> <p>`

use gam_runtime::resource::MemoryGovernor;
use gam_sae::parameter_decomposition::supports::EvidenceStatus;
use gam_sae::response::finite_grid::{FiniteGridResponse, RectangleComplement, centring_factor};
use gam_sae::response::interaction::{PortVariance, total_interactions};
use ndarray::Array2;

fn describe(label: &str, complement: &RectangleComplement) {
    let witness = complement.witness.as_ref();
    let bounds = match &complement.evidence {
        EvidenceStatus::Exact {
            value,
            numerical_error,
            ..
        } => format!("exact {value:.4} ± {numerical_error:.2e}"),
        other => format!(
            "[{:.4}, {:.4}]",
            other.lower_bound().unwrap_or(f64::NAN),
            other.upper_bound().unwrap_or(f64::NAN)
        ),
    };
    println!(
        "{label}: max|Δ| = {:.4}, eps* in {bounds}, max|I_anova| = {:.4}, witness corner {:?} anchor {:?} output {:?}",
        complement.max_difference,
        complement.anova_residual_sup,
        witness.map(|w| w.corner.clone()),
        witness.map(|w| w.anchor),
        witness.map(|w| w.output),
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: modadd_finite_grid <logits.f64> <p>")?;
    let p: usize = args
        .get(2)
        .ok_or("usage: modadd_finite_grid <logits.f64> <p>")?
        .parse()?;
    let bytes = std::fs::read(path)?;
    if bytes.len() != p * p * p * 8 {
        return Err(format!("{path}: {} bytes, expected {}", bytes.len(), p * p * p * 8).into());
    }
    let logits: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect();
    let logits = Array2::from_shape_vec((p * p, p), logits)?;
    let governor = MemoryGovernor::global();
    let factor = centring_factor(p);

    for (name, output_factor) in [("centred", Some(factor.view())), ("raw", None)] {
        let grid = FiniteGridResponse::new(vec![p, p], logits.view(), output_factor, governor)?;
        let screen = total_interactions(&grid)?;
        let total = screen.total_variance();
        let interaction = screen.interaction(0, 1).ok_or("pair")?;
        println!(
            "[{name}] (a,b): V = {:.4} ± {:.1e}; I_ab / V = {:.4}; V(a)/V = {:.4}; V(b)/V = {:.4}; blocks {:?}",
            total.value,
            total.band,
            interaction.value / total.value,
            grid.retained_variance(&[0])?.value / total.value,
            grid.retained_variance(&[1])?.value / total.value,
            screen.additive_blocks()
        );
        describe(
            &format!("[{name}] (a,b) rectangle"),
            &grid.rectangle_complement(0, 1)?,
        );

        let rotated = grid.reindex(
            vec![p, p],
            |cell: &[usize]| vec![(cell[0] + cell[1]) % p, (cell[0] + p - cell[1]) % p],
            governor,
        )?;
        let screen = total_interactions(&rotated)?;
        let s_main = rotated.retained_variance(&[0])?;
        let d_main = rotated.retained_variance(&[1])?;
        let sd = screen.interaction(0, 1).ok_or("pair")?;
        println!(
            "[{name}] (s,d): V(s)/V = {:.4}; V(d)/V = {:.2e}; I_sd/V = {:.4}; blocks {:?}",
            s_main.value / total.value,
            d_main.value / total.value,
            sd.value / total.value,
            screen.additive_blocks()
        );
        describe(
            &format!("[{name}] (s,d) rectangle"),
            &rotated.rectangle_complement(0, 1)?,
        );

        let law = rotated.conditional_mean(&[0])?;
        let mut correct = 0usize;
        for a in 0..p {
            for b in 0..p {
                let s = (a + b) % p;
                let row = law.values.row(s);
                let best = (0..p)
                    .max_by(|&l, &r| row[l].total_cmp(&row[r]))
                    .ok_or("outputs")?;
                correct += usize::from(best == s);
            }
        }
        println!("[{name}] s-only law argmax: {correct} / {} inputs", p * p);
    }
    Ok(())
}

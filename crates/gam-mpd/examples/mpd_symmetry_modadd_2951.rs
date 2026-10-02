//! Symmetry discovery on a trained modular-addition transformer (#2951), with nothing declared.
//!
//! `mpd_symmetry_modadd_2951 EXPORT_DIR`
//!
//! `EXPORT_DIR` is what `bench/mpd_symmetry_export_2951.py modadd` writes from a
//! `bench/mpd_modadd_2951.py` run: raw little-endian float64 tensors `<name>.f64` shaped by `export.json`. The network is that
//! benchmark's one layer read at `=`: `x_j = W_E[t_j] + W_pos[j]` for `(a, b, =)`, causal softmax
//! attention (score scale `1/√d_head`), a ReLU MLP and the unembedding.
//!
//! 1. The token-space operators are the Grams of the tables the network reads per token: `W_E`
//!    and each head's value-output image `W_E W_V,hᵀ W_O,hᵀ`. [`candidate_subdomains`] reports
//!    the cells and twins an exact symmetry could act on (a trained network's symmetry is only
//!    approximate, so its tokens separate at rounding level), and [`discover_permutations`]
//!    finds the gap-certified group on the `p + 1` tokens with no cycle, group or frequency
//!    named. The same tables read up to an affine map give one whitened projector per resolved
//!    leading subspace of the stacked table ([`TokenOperators::leading_subspaces`]); the groups
//!    found there are reported per rank.
//! 2. Each generator is certified on the network function over every `(a, b) ∈ Z_p²` with the
//!    output permutation discovered by assignment ([`certify_on_function`]).
//! 3. [`isotypic_basis`] gives the forced blocks; each is labelled by the frequency `k` read from
//!    its character at the group's canonical cycle (`PermutationGroup::regular_cycle`), and by
//!    its character at the token-order shift `t ↦ t + 1` when the group holds it, so the planes can
//!    be compared with the run card's `W_E` spectrum. Each block's share of `W_E`, the off-block
//!    (Schur-forbidden) fraction of every operator and the generators' code are reported.
//!
//! JSON on stdout.

use gam_linalg::faer_ndarray::fast_abt;
use gam_mpd::symmetry::{
    TokenFunction, TokenOperators, candidate_subdomains, certify_on_function, discover_permutations,
    generator_code_bits, isotypic_basis,
};
use ndarray::{Array2, s};
use serde_json::json;
use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_tensor(dir: &Path, record: &serde_json::Value, name: &str) -> Result<Array2<f64>, String> {
    let shape = record["files"][name]["shape"]
        .as_array()
        .ok_or_else(|| format!("export.json: no shape for {name}"))?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or_else(|| format!("export.json: {name} shape")))
        .collect::<Result<Vec<_>, _>>()?;
    let [rows, cols] = shape[..] else { return Err(format!("{name}: shape {shape:?} is not two axes")) };
    let path = dir.join(format!("{name}.f64"));
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|error| error.to_string())
}

struct Model {
    p: usize,
    heads: usize,
    head_dim: usize,
    w_e: Array2<f64>,
    w_pos: Array2<f64>,
    w_q: Array2<f64>,
    w_k: Array2<f64>,
    w_v: Array2<f64>,
    w_o: Array2<f64>,
    w_in: Array2<f64>,
    b_in: Array2<f64>,
    w_out: Array2<f64>,
    b_out: Array2<f64>,
    w_u: Array2<f64>,
}

impl Model {
    fn load(dir: &Path) -> Result<(Self, serde_json::Value), String> {
        let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| e.to_string())?;
        let record: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let field = |name: &str| record["config"][name].as_u64().map(|v| v as usize).ok_or(format!("config.{name}"));
        let t = |name: &str| read_tensor(dir, &record, name);
        let model = Self {
            p: field("p")?,
            heads: field("n_heads")?,
            head_dim: field("d_head")?,
            w_e: t("W_E")?,
            w_pos: t("W_pos")?,
            w_q: t("W_Q")?,
            w_k: t("W_K")?,
            w_v: t("W_V")?,
            w_o: t("W_O")?,
            w_in: t("W_in")?,
            b_in: t("b_in")?,
            w_out: t("W_out")?,
            b_out: t("b_out")?,
            w_u: t("W_U")?,
        };
        Ok((model, record))
    }

    /// Logits at `=` for every `(a, b)`, row `a·p + b`.
    fn logits(&self) -> Array2<f64> {
        let p = self.p;
        let (heads, dh) = (self.heads, self.head_dim);
        let scale = 1.0 / (dh as f64).sqrt();
        let mut out = Array2::<f64>::zeros((p * p, p));
        let equals = p;
        for a in 0..p {
            for b in 0..p {
                let tokens = [a, b, equals];
                let x = Array2::from_shape_fn((3, self.w_e.ncols()), |(j, c)| self.w_e[[tokens[j], c]] + self.w_pos[[j, c]]);
                let q = fast_abt(&x, &self.w_q);
                let k = fast_abt(&x, &self.w_k);
                let v = fast_abt(&x, &self.w_v);
                let mut z = vec![0.0; heads * dh];
                for h in 0..heads {
                    let range = h * dh..(h + 1) * dh;
                    let scores: Vec<f64> = (0..3)
                        .map(|j| {
                            let (query, key) = (q.slice(s![2, range.clone()]), k.slice(s![j, range.clone()]));
                            query.iter().zip(key.iter()).map(|(a, b)| a * b).sum::<f64>() * scale
                        })
                        .collect();
                    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for (j, w) in weights.iter().enumerate() {
                        for c in range.clone() {
                            z[c] += w / total * v[[j, c]];
                        }
                    }
                }
                let z = Array2::from_shape_vec((1, heads * dh), z).expect("one row");
                let resid = &x.slice(s![2..3, ..]) + &fast_abt(&z, &self.w_o);
                let hidden = (fast_abt(&resid, &self.w_in) + &self.b_in).mapv(|v| v.max(0.0));
                let resid = &resid + &(fast_abt(&hidden, &self.w_out) + &self.b_out);
                let logits = fast_abt(&resid, &self.w_u);
                out.row_mut(a * p + b).assign(&logits.row(0));
            }
        }
        out
    }

    /// `W_E W_V,hᵀ W_O,hᵀ`: what each token writes through head `h`.
    fn ov_image(&self, h: usize) -> Array2<f64> {
        let range = h * self.head_dim..(h + 1) * self.head_dim;
        let w_v = self.w_v.slice(s![range.clone(), ..]).to_owned();
        let w_o = self.w_o.slice(s![.., range]).to_owned();
        fast_abt(&fast_abt(&self.w_e, &w_v), &w_o)
    }
}

/// `tr(Φ_Bᵀ P_g Φ_B) / m`: one copy's character at `g`.
fn character_at(basis: &Array2<f64>, columns: std::ops::Range<usize>, multiplicity: usize, g: &[u32]) -> f64 {
    let block = basis.slice(s![.., columns]);
    let n = block.nrows();
    (0..n).map(|x| block.row(g[x] as usize).dot(&block.row(x))).sum::<f64>() / multiplicity as f64
}

/// The frequency `k ∈ 1..(p−1)/2` whose `2 cos(2πk/p)` is nearest `χ`, with the distance.
fn frequency_of(chi: f64, p: usize) -> (usize, f64) {
    (1..=(p - 1) / 2)
        .map(|k| (k, (2.0 * (TAU * k as f64 / p as f64).cos() - chi).abs()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f64::INFINITY))
}

fn main() -> Result<(), String> {
    let dir = PathBuf::from(std::env::args().nth(1).ok_or("usage: mpd_symmetry_modadd_2951 EXPORT_DIR")?);
    let (model, record) = Model::load(&dir)?;
    let p = model.p;
    let n = p + 1;
    let started = Instant::now();
    let mut tables = vec![model.w_e.clone()];
    for h in 0..model.heads {
        tables.push(model.ov_image(h));
    }
    let views: Vec<_> = tables.iter().map(|t| t.view()).collect();
    let subdomains = candidate_subdomains(&views).map_err(|e| e.to_string())?;
    let ops = TokenOperators::from_tables(&views).map_err(|e| e.to_string())?;
    let discovery = discover_permutations(&ops).map_err(|e| e.to_string())?;
    let discovery_seconds = started.elapsed().as_secs_f64();
    let group = &discovery.group;
    let levels: Vec<_> = discovery
        .levels
        .iter()
        .map(|level| {
            json!({
                "base": level.base,
                "orbit_size": level.orbit.len(),
                "gap": level.gap,
                "max_orbit_defect": level.orbit_defects.iter().copied().fold(0.0, f64::max),
                "lowest_candidates": level.candidates.iter().take(4).collect::<Vec<_>>(),
                "first_rejected": level.gap.as_ref().and_then(|g| g.gamma),
            })
        })
        .collect();
    // The function certificate over every (a, b).
    let logits = model.logits();
    let mut argmax_correct = 0;
    for a in 0..p {
        for b in 0..p {
            let row = logits.row(a * p + b);
            let best = row.iter().enumerate().fold((0, f64::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc }).0;
            if best == (a + b) % p {
                argmax_correct += 1;
            }
        }
    }
    let tokens = Array2::from_shape_fn((p * p, 2), |(r, slot)| if slot == 0 { (r / p) as u32 } else { (r % p) as u32 });
    let function = TokenFunction { tokens: tokens.view(), acted: &[true, true], logits: logits.view(), radii: None };
    let certificates: Vec<_> = group
        .generators()
        .iter()
        .map(|g| match certify_on_function(g, &function) {
            Ok(c) => json!({
                "generator_moves_equals": g[p] as usize != p,
                "rows": c.rows,
                "rows_within_band": c.rows_within_band,
                "kl_bits": c.kl_bits,
                "kl_bits_per_row": c.kl_bits / c.rows as f64,
                "max_row_kl_bits": c.max_row_kl_bits,
                "argmax_agreement": c.argmax_agreement,
                "output_equals_input_permutation": c.output_permutation.iter().enumerate().all(|(x, &y)| g[x] == y),
            }),
            Err(error) => json!({ "refused": error.to_string() }),
        })
        .collect();
    // The forced basis and its labels.
    let basis = isotypic_basis(group).map_err(|e| e.to_string())?;
    let unit_shift: Vec<u32> = (0..n).map(|t| if t < p { ((t + 1) % p) as u32 } else { t as u32 }).collect();
    let holds_unit_shift = group.contains(&unit_shift);
    // The same tables read up to an affine map: every resolved leading subspace of the stacked
    // token table, whitened, and the group its projector keeps.
    let stacked = ndarray::concatenate(ndarray::Axis(1), &views).map_err(|e| e.to_string())?;
    let mut subspaces = Vec::new();
    for (rank, ops) in TokenOperators::leading_subspaces(stacked.view(), 0.0).map_err(|e| e.to_string())? {
        let found = discover_permutations(&ops).map_err(|e| e.to_string())?;
        let order = found.group.order().map_or(f64::INFINITY, |o| o as f64);
        if order <= 1.0 {
            continue;
        }
        let certificate = found.group.generators().first().map(|g| certify_on_function(g, &function));
        subspaces.push(json!({
            "rank": rank,
            "group_order": order,
            "abelian": found.group.is_abelian(),
            "exact": found.exact(),
            "exact_band": found.exact_band,
            "max_generator_defect": found.generator_defects.iter().copied().fold(0.0, f64::max),
            "holds_token_order_shift": found.group.contains(&unit_shift),
            "cycle_on_numbers": found.group.cycle_positions().map(|c| c.iter().flatten().count()),
            "first_generator_kl_bits_per_row": certificate.and_then(|c| c.ok()).map(|c| c.kl_bits / c.rows as f64),
        }));
    }
    let cycle = group.regular_cycle();
    let energy = basis.block_energy(model.w_e.view()).map_err(|e| e.to_string())?;
    let mut blocks: Vec<_> = basis
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let token_order = holds_unit_shift
                .then(|| frequency_of(character_at(&basis.basis, b.columns.clone(), b.multiplicity, &unit_shift), p));
            let canonical = cycle.as_ref().map(|c| frequency_of(character_at(&basis.basis, b.columns.clone(), b.multiplicity, c), p));
            json!({
                "irrep_dim": b.irrep_dim,
                "multiplicity": b.multiplicity,
                "kind": b.kind,
                "w_e_share": energy[i],
                "frequency_token_order": token_order.map(|f| f.0),
                "frequency_token_order_residual": token_order.map(|f| f.1),
                "frequency_canonical_cycle": canonical.map(|f| f.0),
                "projector_error": b.projector_error,
            })
        })
        .collect();
    blocks.sort_by(|a, b| b["w_e_share"].as_f64().unwrap_or(0.0).total_cmp(&a["w_e_share"].as_f64().unwrap_or(0.0)));
    let off_block: Vec<f64> = ops
        .operators
        .iter()
        .map(|a| basis.off_block_fraction(a.view()).unwrap_or(f64::NAN))
        .collect();
    let report = json!({
        "export": dir.display().to_string(),
        "p": p,
        "tokens": n,
        "run": record["run"],
        "argmax_correct": argmax_correct,
        "discovery_seconds": discovery_seconds,
        "exact_cells": subdomains.cells,
        "exact_twins": subdomains.twins,
        "exact_refinement_rounds": subdomains.rounds,
        "levels": levels,
        "group_order": group.order().map(|o| o as f64),
        "group_abelian": group.is_abelian(),
        "generators": group.generators().len(),
        "generator_defects": discovery.generator_defects,
        "max_element_defect": discovery.max_element_defect,
        "exact_band": discovery.exact_band,
        "exact": discovery.exact(),
        "holds_token_order_shift": holds_unit_shift,
        "canonical_cycle_step_at_0": cycle.as_ref().map(|c| c[0]),
        "generator_code_bits": generator_code_bits(group).map_err(|e| e.to_string())?,
        "function_certificates": certificates,
        "isotypic": {
            "blocks": basis.blocks.len(),
            "eigenvalue_band": basis.eigenvalue_band,
            "separation_over_band": basis.separation_over_band,
            "commutant_dim": basis.commutant_dim,
            "attempts": basis.attempts,
            "by_w_e_share": blocks,
        },
        "operator_off_block_fraction": off_block,
        "leading_subspaces_with_symmetry": subspaces,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    println!("{text}");
    Ok(())
}

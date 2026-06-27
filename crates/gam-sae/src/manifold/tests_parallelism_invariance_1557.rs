//! #1557 arrow-Schur parallelism-invariance regression test, split out of
//! `tests.rs` to keep that tracked file under the #780 10k-line gate. Declared
//! as a sibling `#[cfg(test)] mod` in `mod.rs`.

use super::*;
use ndarray::array;
use std::sync::Arc;

/// #1557 regression — the arrow-Schur row assembly now runs each parallel row
/// worker inside a `with_nested_parallel` guard so the tiny per-row / per-block
/// faer matmuls reached transitively pin to `Par::Seq` (avoiding the global
/// Rayon-pool barrier-spin against the outer row fan-out). That routing is a
/// pure parallelism-policy change and MUST be numerically bit-identical:
/// summing a tiny per-row product serially vs. via a fanned pool gives the same
/// IEEE result (same single-product reduction order). This test pins exactly
/// that invariant: assembling the SAE arrow-Schur system with faer's global
/// parallelism set to `Par::Seq` and to `Par::rayon(4)` must yield BYTE-IDENTICAL
/// `gb`, and per-row `gt` / `htt` / `htbeta` blocks. The fixture is sized at the
/// issue's K=8 / p=32 / d=1 shape with n=128 rows (≥ `SAE_LOSS_PARALLEL_ROW_MIN`
/// and the assembly parallel floor) so the row-parallel `into_par_iter`
/// assembly path — the one the guard wraps — is genuinely engaged.
#[test]
pub(crate) fn arrow_schur_assembly_is_faer_parallelism_invariant_1557() {
    use ndarray::Array2;
    let n = 128usize;
    let p = 32usize;
    let k = 8usize;
    let m = 5usize; // periodic basis width (odd; M = 2*harmonics+1 for harmonics=2)
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());

    // Distinct per-atom latent coordinates and decoders so every (gt, htt,
    // htbeta) block carries nontrivial, atom-specific structure.
    let mut atoms = Vec::with_capacity(k);
    let mut coord_blocks = Vec::with_capacity(k);
    for atom_idx in 0..k {
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
            ((row as f64 * 0.013 + atom_idx as f64 * 0.071) % 1.0).fract()
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let decoder = Array2::<f64>::from_shape_fn((m, p), |(i, j)| {
            0.1 * ((i as f64 + 1.0) * 0.3 - (j as f64) * 0.017 + atom_idx as f64 * 0.05).sin()
        });
        let atom = SaeManifoldAtom::new(
            format!("periodic_{atom_idx}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .unwrap()
        .with_basis_evaluator(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords);
    }
    let logits = Array2::<f64>::from_shape_fn((n, k), |(row, col)| {
        0.5 * ((row as f64) * 0.021 + (col as f64) * 0.37).sin()
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        0.05 * ((row as f64) * 0.011 - (col as f64) * 0.023).cos()
    });
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![array![0.9_f64.ln()]; k],
    );

    // Capture the entrant global parallelism so we leave the process state
    // exactly as we found it (other tests in this binary share the global).
    let entry_par = faer::get_global_parallelism();

    let assemble = |term: &mut SaeManifoldTerm, par: faer::Par| {
        faer::set_global_parallelism(par);
        term.assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly must succeed")
    };

    let seq = assemble(&mut term, faer::Par::Seq);
    let par = assemble(&mut term, faer::Par::rayon(4));

    faer::set_global_parallelism(entry_par);

    // Global β gradient: byte-identical.
    assert_eq!(seq.gb.len(), par.gb.len(), "gb length mismatch");
    for (i, (&s, &q)) in seq.gb.iter().zip(par.gb.iter()).enumerate() {
        assert_eq!(
            s.to_bits(),
            q.to_bits(),
            "gb[{i}] not bit-identical across faer parallelism (Seq={s}, rayon={q})"
        );
    }

    // Per-row t-block gradient / Hessian / cross-block: byte-identical.
    assert_eq!(seq.rows.len(), par.rows.len(), "row count mismatch");
    assert_eq!(seq.rows.len(), n, "expected n assembled rows");
    for (row, (rs, rq)) in seq.rows.iter().zip(par.rows.iter()).enumerate() {
        assert_eq!(rs.gt.len(), rq.gt.len(), "row {row} gt len mismatch");
        for (a, (&s, &q)) in rs.gt.iter().zip(rq.gt.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                q.to_bits(),
                "row {row} gt[{a}] not bit-identical (Seq={s}, rayon={q})"
            );
        }
        assert_eq!(rs.htt.dim(), rq.htt.dim(), "row {row} htt dim mismatch");
        for ((i, j), &s) in rs.htt.indexed_iter() {
            let q = rq.htt[[i, j]];
            assert_eq!(
                s.to_bits(),
                q.to_bits(),
                "row {row} htt[{i},{j}] not bit-identical (Seq={s}, rayon={q})"
            );
        }
        assert_eq!(
            rs.htbeta.dim(),
            rq.htbeta.dim(),
            "row {row} htbeta dim mismatch"
        );
        for ((i, j), &s) in rs.htbeta.indexed_iter() {
            let q = rq.htbeta[[i, j]];
            assert_eq!(
                s.to_bits(),
                q.to_bits(),
                "row {row} htbeta[{i},{j}] not bit-identical (Seq={s}, rayon={q})"
            );
        }
    }
}

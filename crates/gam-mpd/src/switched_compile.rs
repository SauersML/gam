//! Static dead-block elimination for a fixed autonomous switching policy.
//!
//! This preserves that policy's execution, not the all-on decomposition or native weights.
//! Only feature-free, unit-free switches with finite nonpositive bias are proved off.
//! Raw reads of off writers are retained when any retained switch reads them. An entirely
//! dead site retains one explicitly off block because masked interfaces require positive width.

use crate::gates::Switch;
use crate::masked::{Library, Masked, Site};
use crate::operator_program::{Node, OperatorProgram};
use ndarray::Axis;

pub struct Compiled {
    pub masked: Masked,
    pub switches: Vec<Vec<Switch>>,
    /// Compiled block/amplitude index to its original index, per site.
    pub original_blocks: Vec<Vec<usize>>,
    pub original_amplitudes: Vec<Vec<usize>>,
    /// Original index to compiled index; None means discarded.
    pub block_map: Vec<Vec<Option<usize>>>,
    pub amplitude_map: Vec<Vec<Option<usize>>>,
    pub counts: Vec<SiteCounts>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteCounts {
    pub original_blocks: usize,
    pub kept_blocks: usize,
    pub original_reads: usize,
    pub kept_reads: usize,
    pub dummy: bool,
}

/// Compile libraries against the native program and its native sites. The caller retains the
/// original policy if it needs interventions that independently override switch decisions.
pub fn compile(
    native: &OperatorProgram,
    sites: Vec<Site>,
    libraries: Vec<Library>,
    ranks: Vec<Vec<usize>>,
    switches: Vec<Vec<Switch>>,
) -> Result<Compiled, String> {
    let n = sites.len();
    if libraries.len() != n || ranks.len() != n || switches.len() != n {
        return Err("compiler site, library, rank and switch counts differ".into());
    }
    let mut owners = Vec::with_capacity(n);
    for site in 0..n {
        let total = ranks[site]
            .iter()
            .try_fold(0usize, |sum, rank| sum.checked_add(*rank));
        if ranks[site].is_empty()
            || ranks[site].contains(&0)
            || total != Some(libraries[site].v.nrows())
            || libraries[site].u.nrows() != libraries[site].v.nrows()
            || switches[site].len() != ranks[site].len()
        {
            return Err(format!(
                "site {site}: invalid block partition or library rows"
            ));
        }
        if libraries[site]
            .v
            .iter()
            .chain(libraries[site].u.iter())
            .chain(libraries[site].mean.iter())
            .any(|x| !x.is_finite())
        {
            return Err(format!("site {site}: nonfinite library"));
        }
        owners.push(
            ranks[site]
                .iter()
                .enumerate()
                .flat_map(|(block, rank)| std::iter::repeat_n(block, *rank))
                .collect::<Vec<_>>(),
        );
    }
    // Validate the full policy, including switches that will be removed. Use the same insertion
    // order as Masked::build_blocks for availability checks (multiple sites can share a writer).
    let order = |site: usize| {
        (
            sites[site]
                .writes
                .iter()
                .copied()
                .min()
                .unwrap_or(usize::MAX),
            site,
        )
    };
    for (site, row) in switches.iter().enumerate() {
        for switch in row {
            let d = switch.features.len();
            if switch.linear.len() != d
                || !switch.beta.is_finite()
                || switch.linear.iter().any(|x| !x.is_finite())
                || switch.units.iter().any(|u| {
                    u.w.len() != d
                        || !u.d.is_finite()
                        || !u.c.is_finite()
                        || u.w.iter().any(|x| !x.is_finite())
                })
            {
                return Err(format!("site {site}: invalid switch coefficients"));
            }
            for f in &switch.features {
                if f.site >= n || f.piece >= owners[f.site].len() || f.lag > 1 {
                    return Err(format!("site {site}: invalid feature"));
                }
                if order(f.site) > order(site) {
                    return Err(format!("site {site}: unavailable feature"));
                }
                if f.lag == 1 {
                    let (start, end) = (order(f.site).0, order(site).0);
                    if start >= end
                        || !native.nodes.get(start..end).is_some_and(|nodes| {
                            nodes.iter().any(|node| {
                                matches!(node, Node::Attend { causal: true, .. } | Node::Mix { .. })
                            })
                        })
                    {
                        return Err(format!("site {site}: lagged feature lacks causal mixing"));
                    }
                }
            }
        }
    }
    let mut kept: Vec<Vec<bool>> = switches
        .iter()
        .map(|row| {
            row.iter()
                .map(|s| !(s.features.is_empty() && s.units.is_empty() && s.beta <= 0.0))
                .collect()
        })
        .collect();
    loop {
        let mut changed = false;
        for (site, row) in switches.iter().enumerate() {
            for (block, switch) in row.iter().enumerate() {
                if !kept[site][block] {
                    continue;
                }
                for f in &switch.features {
                    let owner = owners[f.site][f.piece];
                    if !kept[f.site][owner] {
                        kept[f.site][owner] = true;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut counts = Vec::with_capacity(n);
    for site in 0..n {
        let dummy = !kept[site].iter().any(|x| *x);
        if dummy {
            kept[site][0] = true;
        }
        counts.push(SiteCounts {
            original_blocks: ranks[site].len(),
            kept_blocks: kept[site].iter().filter(|x| **x).count(),
            original_reads: owners[site].len(),
            kept_reads: owners[site]
                .iter()
                .filter(|block| kept[site][**block])
                .count(),
            dummy,
        });
    }
    let original_blocks: Vec<Vec<usize>> = kept
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter_map(|(i, keep)| keep.then_some(i))
                .collect()
        })
        .collect();
    let original_amplitudes: Vec<Vec<usize>> = owners
        .iter()
        .enumerate()
        .map(|(site, row)| {
            row.iter()
                .enumerate()
                .filter_map(|(i, block)| kept[site][*block].then_some(i))
                .collect()
        })
        .collect();
    let inverse = |original: &[usize], width: usize| {
        let mut map = vec![None; width];
        for (new, old) in original.iter().enumerate() {
            map[*old] = Some(new);
        }
        map
    };
    let block_map = (0..n)
        .map(|s| inverse(&original_blocks[s], ranks[s].len()))
        .collect();
    let amplitude_map: Vec<_> = (0..n)
        .map(|s| inverse(&original_amplitudes[s], owners[s].len()))
        .collect();
    let remapped = (0..n)
        .map(|site| {
            original_blocks[site]
                .iter()
                .map(|block| {
                    let mut switch = switches[site][*block].clone();
                    for f in &mut switch.features {
                        f.piece =
                            amplitude_map[f.site][f.piece].expect("feature closure retains read");
                    }
                    switch
                })
                .collect()
        })
        .collect();
    let reduced = libraries
        .into_iter()
        .enumerate()
        .map(|(site, library)| Library {
            v: library.v.select(Axis(0), &original_amplitudes[site]),
            u: library.u.select(Axis(0), &original_amplitudes[site]),
            mean: library.mean,
        })
        .collect();
    let reduced_ranks = (0..n)
        .map(|site| {
            original_blocks[site]
                .iter()
                .map(|block| ranks[site][*block])
                .collect()
        })
        .collect();
    let masked = Masked::build_blocks(native, sites, reduced, reduced_ranks)?;
    Ok(Compiled {
        masked,
        switches: remapped,
        original_blocks,
        original_amplitudes,
        block_map,
        amplitude_map,
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Feature;
    use crate::operator_program::{
        Coefficient, Declarations, FamilyInputs, Interface, Operator, Provenance, Slot, SlotValues,
    };
    use crate::precision::DeclaredPrecision;
    use ndarray::{Array1, array};
    use std::sync::Arc;

    fn constant(beta: f64) -> Switch {
        Switch {
            features: vec![],
            beta,
            linear: vec![],
            units: vec![],
            precision: 0,
            function_bits: 0.0,
            listing_bits: 0.0,
            inputs: 0,
        }
    }

    fn fixture() -> (
        OperatorProgram,
        Vec<Site>,
        Vec<Library>,
        Vec<Vec<usize>>,
        Vec<Vec<Switch>>,
    ) {
        let interface = Interface::native(1).unwrap();
        let op = |name: &str| {
            Arc::new(
                Operator::dense(
                    name,
                    interface.clone(),
                    interface.clone(),
                    array![[1.0]],
                    DeclaredPrecision::new(40).unwrap(),
                    Provenance::default(),
                )
                .unwrap(),
            )
        };
        let native = OperatorProgram {
            declarations: Declarations {
                parameters: 1,
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
            },
            bases: vec![],
            operators: vec![op("first"), op("second")],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Gain {
                    input: 0,
                    coefficient: Coefficient::Parameter(0),
                },
                Node::Affine {
                    terms: vec![(1, 0)],
                    bias: None,
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: None,
                },
            ],
            output: 3,
        };
        let all = crate::masked::sites(&native);
        let sites = ["first", "second"]
            .iter()
            .map(|name| all.iter().find(|site| site.name == *name).unwrap().clone())
            .collect();
        let library = || Library {
            v: array![[1.0], [2.0], [3.0], [4.0]],
            u: array![[0.1], [0.2], [0.3], [0.4]],
            mean: Array1::zeros(1),
        };
        let dynamic = Switch {
            features: vec![Feature {
                site: 0,
                piece: 2,
                lag: 0,
                magnitude: true,
            }],
            linear: vec![1.0],
            ..constant(-2.0)
        };
        (
            native,
            sites,
            vec![library(), library()],
            vec![vec![1, 2, 1]; 2],
            vec![
                vec![constant(1.0), constant(-1.0), constant(0.0)],
                vec![constant(-1.0), dynamic, constant(-1.0)],
            ],
        )
    }

    #[test]
    fn retains_rank_two_off_writer_read_and_matches_controls() {
        let (native, sites, libraries, ranks, switches) = fixture();
        let full =
            Masked::build_blocks(&native, sites.clone(), libraries.clone(), ranks.clone()).unwrap();
        let compiled = compile(&native, sites, libraries, ranks, switches.clone()).unwrap();
        assert_eq!(compiled.original_blocks, vec![vec![0, 1], vec![1]]);
        assert_eq!(
            compiled.original_amplitudes,
            vec![vec![0, 1, 2], vec![1, 2]]
        );
        assert_eq!(compiled.switches[1][0].features[0].piece, 2);
        assert_eq!(compiled.block_map[0], vec![Some(0), Some(1), None]);
        for gain in [0.0, 0.5, 1.0, -2.0] {
            let base = FamilyInputs {
                rows: 3,
                slots: vec![SlotValues::Raw(array![[0.0], [1.0], [-3.0]])],
                layout: None,
            };
            let (before, before_masks) =
                crate::switched::execute_at(&full, &base, &switches, &[gain]).unwrap();
            let (after, after_masks) =
                crate::switched::execute_at(&compiled.masked, &base, &compiled.switches, &[gain])
                    .unwrap();
            assert_eq!(
                before.values[full.program.output],
                after.values[compiled.masked.program.output]
            );
            for site in 0..2 {
                assert_eq!(
                    before_masks[site].select(Axis(1), &compiled.original_blocks[site]),
                    after_masks[site]
                );
            }
        }
    }

    #[test]
    fn remaps_amplitude_columns_and_keeps_explicit_dead_site() {
        let (native, sites, libraries, ranks, mut switches) = fixture();
        switches[0] = vec![constant(-1.0); 3];
        switches[1][1].features[0].site = 1;
        let full =
            Masked::build_blocks(&native, sites.clone(), libraries.clone(), ranks.clone()).unwrap();
        let compiled = compile(&native, sites, libraries, ranks, switches.clone()).unwrap();
        assert!(compiled.counts[0].dummy);
        assert_eq!(compiled.original_blocks[0], vec![0]);
        assert_eq!(compiled.switches[1][0].features[0].piece, 1);
        let base = FamilyInputs {
            rows: 1,
            slots: vec![SlotValues::Raw(array![[10.0]])],
            layout: None,
        };
        let (before, _) = crate::switched::execute(&full, &base, &switches).unwrap();
        let (after, _) =
            crate::switched::execute(&compiled.masked, &base, &compiled.switches).unwrap();
        assert_eq!(
            before.values[full.program.output],
            after.values[compiled.masked.program.output]
        );
    }

    #[test]
    fn rejects_malformed_removed_switches_and_partitions() {
        let (native, sites, libraries, ranks, mut switches) = fixture();
        switches[0][2].beta = f64::NAN;
        assert!(
            compile(
                &native,
                sites.clone(),
                libraries.clone(),
                ranks.clone(),
                switches
            )
            .is_err()
        );
        let (_, _, _, _, mut switches) = fixture();
        switches[0][2].features.push(Feature {
            site: 99,
            piece: 0,
            lag: 0,
            magnitude: false,
        });
        switches[0][2].linear.push(0.0);
        assert!(compile(&native, sites.clone(), libraries.clone(), ranks, switches).is_err());
        let (_, _, _, _, switches) = fixture();
        assert!(
            compile(
                &native,
                sites,
                libraries,
                vec![vec![usize::MAX, 2], vec![1, 2, 1]],
                switches
            )
            .is_err()
        );
    }
}

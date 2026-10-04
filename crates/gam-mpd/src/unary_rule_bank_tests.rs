use super::*;
use crate::{
    acceptance::{CostCache, structural_cost},
    operator_program::{Declarations, Interface, Node, OperatorProgram, Provenance, Slot},
};
use ndarray::{Array1, array};
fn fixture() -> Artifact {
    let d = Interface::native(2).expect("valid unary-bank fixture or candidate");
    let dense = |name, x: Array2<f64>| {
        Arc::new(
            Operator::dense(
                name,
                d.clone(),
                d.clone(),
                x.clone(),
                exact_precision(x.iter().copied()).expect("valid unary-bank fixture or candidate"),
                Provenance::default(),
            )
            .expect("valid unary-bank fixture or candidate"),
        )
    };
    let p = OperatorProgram {
        declarations: Declarations {
            parameters: 0,
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
        },
        bases: vec![],
        rules: vec![],
        operators: vec![
            dense("source", array![[1., 2.], [3., 4.]]),
            dense("target0", array![[2., 4.], [6., 8.]]),
            dense("target1", array![[4., 3.], [2., 1.]]),
            Arc::new(
                Operator::diag(
                    "global_gain",
                    d.clone(),
                    Array1::from_vec(vec![1., 2.]),
                    exact_precision([1., 2.]).expect("valid unary-bank fixture or candidate"),
                    Provenance::default(),
                )
                .expect("valid unary-bank fixture or candidate"),
            ),
            Arc::new(Operator::identity("fixed_identity", d)),
        ],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(0, 1)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(0, 2)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(2, 4), (3, 4)],
                bias: None,
            },
        ],
        output: 4,
    };
    Artifact::native(&p)
        .expect("valid unary-bank fixture or candidate")
        .f32_literals()
        .expect("valid unary-bank fixture or candidate")
}
fn limits() -> Limits {
    Limits {
        max_prefixes: 10000,
        max_bodies: 1000,
        max_candidates_including_native: 1000,
        cache_bytes: 1 << 20,
        matrix_workspace_bytes: 1 << 20,
    }
}
fn targets() -> Vec<TargetBinding> {
    (0..2)
        .map(|h| TargetBinding {
            operator: h + 1,
            native_layer: 27,
            head: h,
            reads: vec![0],
            write: 4,
        })
        .collect()
}
#[test]
fn unary_bank_complete_uniform_inventory_and_cached_decoder_parity() {
    let base = fixture();
    let mut bank = UnaryRuleBank::new(&base, targets(), limits())
        .expect("valid unary-bank fixture or candidate");
    assert_eq!(bank.candidate_count, 27);
    assert_eq!(bank.inventory.sources.len(), 4);
    let choices: Vec<_> = bank.choices().collect();
    assert_eq!(choices.len(), 26);
    for choice in choices {
        let target = bank
            .target(choice)
            .expect("valid unary-bank fixture or candidate")
            .clone();
        assert_ne!(target.operator, choice.source);
        let skeleton = bank
            .priced_skeleton(choice)
            .expect("valid unary-bank fixture or candidate");
        let (cached, diagnostic) = bank
            .candidate(choice)
            .expect("valid unary-bank fixture or candidate");
        assert_eq!(
            structural_cost(&skeleton, &mut CostCache::default())
                .expect("valid unary-bank fixture or candidate"),
            structural_cost(&cached, &mut CostCache::default())
                .expect("valid unary-bank fixture or candidate")
        );
        let law = OperatorLaw::Expression {
            body: Arc::new(
                bank.body(choice)
                    .expect("valid unary-bank fixture or candidate")
                    .clone(),
            ),
            sources: vec![choice.source],
        };
        let ordinary = base
            .derive(target.operator, law, diagnostic.amplitude, vec![])
            .expect("valid unary-bank fixture or candidate")
            .bind(
                &format!(
                    "generic unary attention {} post-residual boundary",
                    target.native_layer
                ),
                &target.reads,
                target.write,
            )
            .expect("valid unary-bank fixture or candidate");
        assert_eq!(cached, ordinary);
        let bytes = cached
            .to_bytes()
            .expect("valid unary-bank fixture or candidate");
        let decoded = Artifact::from_bytes(&bytes, &base.program.declarations)
            .expect("valid unary-bank fixture or candidate");
        assert_eq!(
            decoded.program.operators[target.operator].matrix(),
            cached.program.operators[target.operator].matrix()
        );
        assert_eq!(
            decoded
                .to_bytes()
                .expect("valid unary-bank fixture or candidate"),
            bytes
        );
        assert_eq!(
            structural_cost(&cached, &mut CostCache::default())
                .expect("valid unary-bank fixture or candidate"),
            structural_cost(&decoded, &mut CostCache::default())
                .expect("valid unary-bank fixture or candidate")
        );
        assert_eq!(cached.places, base.places);
    }
    assert!(bank.cache_stats.hits > 0);
}
#[test]
fn unary_bank_resource_cases_are_explicit_unresolved() {
    let base = fixture();
    let mut limited = limits();
    limited.max_candidates_including_native = 26;
    assert!(UnaryRuleBank::new(&base, targets(), limited).is_err());
    limited = limits();
    limited.cache_bytes = 0;
    let mut bank = UnaryRuleBank::new(&base, targets(), limited)
        .expect("valid unary-bank fixture or candidate");
    let choice = bank
        .choices()
        .next()
        .expect("valid unary-bank fixture or candidate");
    assert!(bank.candidate(choice).unwrap_err().contains("unresolved"));
    assert_eq!(bank.candidate_count, 27);
    limited = limits();
    limited.matrix_workspace_bytes = 0;
    let mut bank = UnaryRuleBank::new(&base, targets(), limited)
        .expect("valid unary-bank fixture or candidate");
    let choice = bank
        .choices()
        .next()
        .expect("valid unary-bank fixture or candidate");
    assert!(bank.candidate(choice).unwrap_err().contains("unresolved"));
}
#[test]
fn unary_bank_undefined_reciprocal_retained_and_no_fit_rank_cut() {
    let mut base = fixture();
    let old = &base.program.operators[0];
    base.program.operators[0] = Arc::new(
        Operator::dense(
            old.name.clone(),
            old.rows.clone(),
            old.cols.clone(),
            array![[0., 2.], [3., 4.]],
            exact_precision([0., 2., 3., 4.]).expect("valid unary-bank fixture or candidate"),
            old.provenance.clone(),
        )
        .expect("valid unary-bank fixture or candidate"),
    );
    let mut bank = UnaryRuleBank::new(&base, targets(), limits())
        .expect("valid unary-bank fixture or candidate");
    let choices: Vec<_> = bank.choices().collect();
    assert_eq!(choices.len(), 26);
    let mut undefined = 0;
    for choice in choices {
        if bank.candidate(choice).is_err() {
            undefined += 1;
        }
    }
    assert_eq!(undefined, 4); // Reciprocal and self-divide for source0, for both targets.
}

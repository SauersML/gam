use super::*;
use crate::acceptance::{assess_once_local_first, StagedAssessment, EpisodeScore};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountRun<'a> { inner: FamilyRun<'a>, calls: AtomicUsize }
impl RunCheck for CountRun<'_> {
    fn episodes(&self, artifact: &Artifact) -> Result<Vec<EpisodeScore>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.episodes(artifact)
    }
}

#[test]
fn local_first_rejection_never_runs_and_is_scoped_to_declared_grid() {
    let width = native(1); let classes = native(2);
    // A zero readout makes Run exactly indifferent to the wrong internal gain.
    let model = raw_program(1, vec![Operator::identity("I", width.clone()),
        dense("zero-head", &classes, &width, array![[0.0], [0.0]])],
        vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None }]);
    let family = grid(&[-1.0, 1.0], 1);
    let candidate = Artifact::native(&model).unwrap().replace_block("gain",
        Callee::New(rule("gain", vec![width.clone()], vec![Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, 2)], bias: None }])),
        vec![Argument::Native(0)], 1,
        vec![dense("gain", &width, &width, array![[2.0]])]).unwrap();
    let bytes = candidate.to_bytes().unwrap();
    let replay = Artifact::from_bytes(&bytes, &model.declarations).unwrap();
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    let local = Local::new(&model, family.clone(), None, 64);
    let run = CountRun { inner: clean_run(&model, &family), calls: AtomicUsize::new(0) };
    let narrow = Constraint { local: 0.1, run: 1e-6 };
    let widest = Constraint { local: 0.5, run: 1e-6 };
    let staged = assess_once_local_first(&local, &run, &replay, &[narrow, widest], &mut CostCache::default()).unwrap();
    assert!(matches!(staged, StagedAssessment::LocalRejected { .. }));
    assert!(staged.run_measure().is_none());
    assert_eq!(run.calls.load(Ordering::SeqCst), 0);
    let codec = crate::operator_program::NativeOperatorCodec::new(&model, 1 << 20).unwrap();
    let cached = crate::acceptance::assess_once_local_first_with_native_codec(
        &local, &run, &replay, &[narrow, widest], &mut CostCache::default(), &codec).unwrap();
    assert_eq!(run.calls.load(Ordering::SeqCst), 0);
    let StagedAssessment::LocalRejected { cost: ac, local: al, local_measure: am, max_local_tolerance: at } = &staged else { panic!("ordinary Run measured") };
    let StagedAssessment::LocalRejected { cost: bc, local: bl, local_measure: bm, max_local_tolerance: bt } = &cached else { panic!("cached Run measured") };
    assert_eq!((ac, al, am, at), (bc, bl, bm, bt));
    assert_eq!(staged.verdict(narrow).unwrap(), FidelityVerdict::Violates);
    assert_eq!(staged.verdict(widest).unwrap(), FidelityVerdict::Violates);
    assert_eq!(staged.verdict(Constraint { local: 0.75, run: 1e-6 }).unwrap(), FidelityVerdict::Unresolved);
    let full = crate::acceptance::assess_once(&local, &run, &replay, narrow, &mut CostCache::default()).unwrap();
    assert_eq!(full.run.verdict(), FidelityVerdict::Meets);
    assert_eq!(staged.cost(), full.cost);
    // Narrow rejection cannot prune a candidate that passes the declared widest delta.
    let broad = Constraint { local: 2.0, run: 1e-6 };
    let kept = assess_once_local_first(&local, &run, &replay, &[narrow, broad], &mut CostCache::default()).unwrap();
    assert!(matches!(kept, StagedAssessment::Complete(_)));
    assert_eq!(kept.verdict(narrow).unwrap(), FidelityVerdict::Violates);
    assert_eq!(kept.verdict(broad).unwrap(), FidelityVerdict::Meets);
    let ordinary = crate::acceptance::assess_once(&local, &run, &replay, broad, &mut CostCache::default()).unwrap();
    let StagedAssessment::Complete(ref completed) = kept else { panic!("widest passing candidate was screened") };
    assert_eq!(completed.cost, ordinary.cost);
    assert_eq!(completed.local, ordinary.local);
    assert_eq!(completed.local_measure, ordinary.local_measure);
    assert_eq!(completed.run, ordinary.run);
    assert_eq!(completed.run_measure, ordinary.run_measure);
    let cached = crate::acceptance::assess_once_local_first_with_native_codec(
        &local, &run, &replay, &[narrow, broad], &mut CostCache::default(), &codec).unwrap();
    let StagedAssessment::Complete(cached) = cached else { panic!("cached widest candidate screened") };
    assert_eq!(cached.cost, ordinary.cost);
    assert_eq!(cached.local, ordinary.local);
    assert_eq!(cached.local_measure, ordinary.local_measure);
    assert_eq!(cached.run, ordinary.run);
    assert_eq!(cached.run_measure, ordinary.run_measure);
    let native_artifact = Artifact::native(&model).unwrap();
    let cached_native = crate::acceptance::assess_once_local_first_with_native_codec(
        &local, &run, &native_artifact, &[broad], &mut CostCache::default(), &codec).unwrap();
    assert_eq!(cached_native.verdict(broad).unwrap(), FidelityVerdict::Meets);
    assert!(codec.usage().encoded_native_operator_hits > 0);
    assert!(codec.usage().decoded_native_operator_hits > 0);
    // A tolerance inside the genuine comparison interval remains unresolved and runs.
    let boundary = full.local_measure.worst().unwrap().worst;
    assert_eq!(full.local.with_tolerance(boundary).unwrap().verdict(), FidelityVerdict::Unresolved);
    let before = run.calls.load(Ordering::SeqCst);
    let ambiguous = assess_once_local_first(&local, &run, &replay,
        &[Constraint { local: boundary, run: 1e-6 }], &mut CostCache::default()).unwrap();
    assert!(matches!(ambiguous, StagedAssessment::Complete(_)));
    assert_eq!(run.calls.load(Ordering::SeqCst), before + 1);
    for grid in [vec![], vec![Constraint { local: f64::NAN, run: 1e-6 }], vec![Constraint { local: 0.0, run: -1.0 }]] {
        assert!(assess_once_local_first(&local, &run, &replay, &grid, &mut CostCache::default()).is_err());
    }
}

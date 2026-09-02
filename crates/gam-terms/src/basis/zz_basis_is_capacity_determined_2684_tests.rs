//! #2684: a model's basis is a function of `(data, spec)`, never of how busy
//! the machine was when it was built.
//!
//! The filed incident: identical inputs built different bases in different
//! processes, and one of them refused a 300x12 design — 28,800 bytes — on a
//! node with hundreds of GB free. The mechanism was that the
//! single-materialization ceiling was `3/4 x FREE memory`, and a cgroup sitting
//! at its hard limit (the state any cgroup settles into once it has touched the
//! page cache) reports almost nothing free. The ceiling therefore slid
//! continuously toward zero while the job's actual capacity never moved, and
//! four processes on one node computed four different routing thresholds — and,
//! because that threshold was hashed into the Duchon basis fingerprint, four
//! different basis identities.
//!
//! The two questions had been collapsed onto one number:
//!
//! * "could a dense footprint this large ever live in this process?" is a
//!   question about CAPACITY, it decides a route, and its answer must be
//!   stationary for the run;
//! * "does this allocation fit right now?" is a question about FREE memory, and
//!   the ledger's `try_reserve` answers it with typed, routable evidence.
//!
//! What follows asserts the separation end to end, on the shipped code: two
//! observations of one cgroup that differ ONLY in how much of it was charged at
//! the instant of the probe must produce the same policy, and the same basis.

use super::types::should_use_lazy_spatial_design;
use super::*;
use gam_runtime::resource::ResourcePolicy;
use gam_runtime::test_support::simulated_cgroup_memory_environment;
use ndarray::Array2;

/// The design the issue reports being refused: 300 rows x 12 f64 columns.
const REPORTED_DESIGN_BYTES: usize = 300 * 12 * std::mem::size_of::<f64>();

/// Host numbers measured on the MSI compute node in the report: a `--mem=6g`
/// job on a box with ~448 GB free of ~503 GiB total.
const HOST_AVAILABLE_BYTES: u64 = 448_648_040_448;
const HOST_TOTAL_BYTES: u64 = 527_799_400 * 1024;
const JOB_LIMIT_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// 300 points in 2-D, deterministic so the fingerprint is fixed across runs and
/// the two arms are provably handed the SAME data (they share this array).
fn fixture_data() -> Array2<f64> {
    let rows = 300;
    let mut data = Array2::<f64>::zeros((rows, 2));
    for i in 0..rows {
        let t = i as f64 / (rows as f64 - 1.0);
        // A second coordinate that is not a function of the first, so the
        // center selection has a genuinely two-dimensional cloud to work on.
        let u = ((i * 7) % rows) as f64 / (rows as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = u;
    }
    data
}

/// 12 centers, so the kernel block is 300x12 = the reported footprint exactly.
fn fixture_spec() -> DuchonBasisSpec {
    DuchonBasisSpec {
        center_strategy: CenterStrategy::EqualMass { num_centers: 12 },
        periodic: None,
        length_scale: None,
        power: 0.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: None,
    }
}

fn build_under(policy: ResourcePolicy) -> BasisBuildResult {
    let data = fixture_data();
    let spec = fixture_spec();
    let mut workspace = BasisWorkspace::with_policy(policy);
    build_duchon_basiswithworkspace(data.view(), &spec, &mut workspace)
        .expect("the 300x12 Duchon fixture must build; refusing it is the defect under test")
}

#[test]
fn one_cgroup_read_at_two_load_levels_yields_one_policy_and_one_basis_2684() {
    // ONE job — same host, same hard limit — observed at two instants.
    let idle = simulated_cgroup_memory_environment(
        HOST_AVAILABLE_BYTES,
        HOST_TOTAL_BYTES,
        JOB_LIMIT_BYTES,
        92_827_648,
    );
    let pinned = simulated_cgroup_memory_environment(
        HOST_AVAILABLE_BYTES,
        HOST_TOTAL_BYTES,
        JOB_LIMIT_BYTES,
        JOB_LIMIT_BYTES - 4_096,
    );

    // The two readings are the two readings the sampler actually saw: an idle
    // cgroup with the whole ceiling free, and one four kilobytes from its
    // limit. If these were not different the rest of this test would be
    // asserting nothing.
    assert!(
        idle.available_bytes() > 6_000_000_000,
        "idle arm must have the ceiling free, got {}",
        idle.available_bytes()
    );
    assert_eq!(pinned.available_bytes(), 4_096);
    // Capacity, by contrast, is the same number in both — that is the whole
    // claim, so it is asserted rather than assumed.
    assert_eq!(idle.capacity_bytes(), pinned.capacity_bytes());
    assert_eq!(idle.capacity_bytes(), JOB_LIMIT_BYTES);

    let idle_policy = ResourcePolicy::for_observed_memory(&idle);
    let pinned_policy = ResourcePolicy::for_observed_memory(&pinned);

    // NON-VACUITY. The pinned reading is one that an availability-denominated
    // ceiling could not have carried this design under: 3/4 of 4,096 bytes is
    // 3,072, an order of magnitude below the 28,800-byte design. So if the two
    // policies below agree, the ceiling is demonstrably not being read off free
    // memory — the assertion had a way to fail and did not take it.
    let availability_denominated_cap = (pinned.available_bytes() as usize) / 4 * 3;
    assert!(
        availability_denominated_cap < REPORTED_DESIGN_BYTES,
        "the pinned arm must be a reading under which a free-memory ceiling would refuse the \
         design, or this test cannot detect the defect: cap-from-free={availability_denominated_cap}, \
         design={REPORTED_DESIGN_BYTES}"
    );
    assert_eq!(
        idle_policy.max_single_materialization_bytes,
        pinned_policy.max_single_materialization_bytes,
        "the materialization ceiling moved with ambient load"
    );
    assert!(
        pinned_policy.max_single_materialization_bytes > REPORTED_DESIGN_BYTES,
        "a 6 GiB job must admit a 28,800-byte design however busy it is, got a cap of {}",
        pinned_policy.max_single_materialization_bytes
    );

    // And therefore: one basis. Same shape, same columns bit for bit, same
    // penalties bit for bit.
    let from_idle = build_under(idle_policy);
    let from_pinned = build_under(pinned_policy);

    assert_eq!(
        (from_idle.design.nrows(), from_idle.design.ncols()),
        (from_pinned.design.nrows(), from_pinned.design.ncols()),
        "the basis changed SHAPE with ambient load"
    );
    let idle_columns = from_idle
        .design
        .as_dense_ref()
        .expect("the 6 GiB-capacity arm routes dense")
        .clone();
    let pinned_columns = from_pinned
        .design
        .as_dense_ref()
        .expect("the 6 GiB-capacity arm routes dense under load too");
    assert_eq!(
        idle_columns.dim(),
        pinned_columns.dim(),
        "dense design shapes disagree"
    );
    for (row, (left, right)) in idle_columns.iter().zip(pinned_columns.iter()).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "design element {row} differs between the idle and the loaded process: {left} vs {right}"
        );
    }
    assert_eq!(
        from_idle.active_penalties.len(),
        from_pinned.active_penalties.len(),
        "penalty count changed with ambient load"
    );
    for (index, (left, right)) in from_idle
        .active_penalties
        .iter()
        .zip(from_pinned.active_penalties.iter())
        .enumerate()
    {
        assert_eq!(
            left.matrix.dim(),
            right.matrix.dim(),
            "penalty {index} changed shape with ambient load"
        );
        for (cell, (a, b)) in left.matrix.iter().zip(right.matrix.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "penalty {index} cell {cell} differs between the idle and the loaded process"
            );
        }
    }
}

#[test]
fn the_ceiling_still_refuses_what_capacity_cannot_hold_2684() {
    // The mirror image of the test above, and the reason it is not vacuous: a
    // ceiling that never says no would satisfy every assertion there. This job
    // has a genuinely tiny CAPACITY — a whole cgroup of 4 KiB — and the route
    // for the same 300x12 design must flip to the streamed one. Note the
    // cgroup here is IDLE: nothing is charged against it, so only capacity can
    // produce the refusal.
    let tiny = simulated_cgroup_memory_environment(
        HOST_AVAILABLE_BYTES,
        HOST_TOTAL_BYTES,
        4_096,
        0,
    );
    assert_eq!(tiny.capacity_bytes(), 4_096);
    assert_eq!(tiny.available_bytes(), 4_096);
    let tiny_policy = ResourcePolicy::for_observed_memory(&tiny);
    assert!(
        tiny_policy.max_single_materialization_bytes < REPORTED_DESIGN_BYTES,
        "a 4 KiB cgroup must not admit a 28,800-byte dense design"
    );
    assert!(
        should_use_lazy_spatial_design(300, 12, &tiny_policy),
        "the 300x12 design must route streamed when capacity cannot hold it"
    );

    let roomy = simulated_cgroup_memory_environment(
        HOST_AVAILABLE_BYTES,
        HOST_TOTAL_BYTES,
        JOB_LIMIT_BYTES,
        JOB_LIMIT_BYTES - 4_096,
    );
    let roomy_policy = ResourcePolicy::for_observed_memory(&roomy);
    assert!(
        !should_use_lazy_spatial_design(300, 12, &roomy_policy),
        "the same design under a 6 GiB ceiling must route dense, however loaded the cgroup is"
    );
}


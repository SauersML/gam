//! Explicit CUDA head parity on an existing fixed spec and serialized artifact list.
//! EXPORT SPEC OUT_JSON TOLERANCE ARTIFACT...; no GPU-free CI assertion.
use gam_mpd::acceptance::{RunCheck, RunMeasure};
use gam_mpd::artifact::Artifact;
use gam_mpd::counterfactual::{Decoder, Spec, passages};
use gam_mpd::import::import_language_model;
use gam_mpd::native_readout::{Budget, Resident};
use gam_mpd::run_check::{LanguageRun, split_sites};
use serde_json::json;
use std::path::Path;

fn main() -> Result<(), String> {
    let mut a: Vec<String> = std::env::args().collect();
    let metric_only = a.get(1).is_some_and(|s| s == "--metric-proposals-only");
    if metric_only { a.remove(1); }
    if a.len() < 6 {
        return Err("EXPORT SPEC OUT_JSON ABSOLUTE_PARITY_TOLERANCE ARTIFACT...".into());
    }
    let tolerance = a[4].parse::<f64>().map_err(|e| e.to_string())?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err("finite nonnegative parity tolerance required".into());
    }
    let decoder = Decoder::from_export(Path::new(&a[1]))?;
    let spec = Spec::load(Path::new(&a[2]), &decoder)?;
    let passages = passages(Path::new(&a[1]), spec.rows)?;
    let native = split_sites(&import_language_model(Path::new(&a[1]), 1, 1)?.program)?;
    let device = gam_gpu::tensor::Device::accelerator(gam_gpu::GpuPolicy::Required)
        .map_err(|e| e.to_string())?
        .ok_or("required CUDA unavailable")?;
    let budget = Budget {
        resident_bytes: 512 << 20,
        workspace_bytes: 256 << 20,
    };
    let head = Resident::new(device.clone(), &decoder, budget)?;
    let width = decoder.embedding().ncols();
    let mut synthetic_max = 0.0_f64;
    let mut synthetic = Vec::new();
    for scale in [0.0, 1e-100, 1.0, 1e100, 1e150] {
        let x = ndarray::Array2::from_shape_fn((2, width), |(r, c)| {
            scale * ((c * 17 + r * 3) % 23) as f64 / 23.0
        });
        let cpu = decoder.log_probs(&x);
        let gpu = head.log_probs(&x)?;
        if cpu.iter().chain(gpu.iter()).any(|x| !x.is_finite()) {
            return Err("nonfinite synthetic log probabilities".into());
        }
        let error = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max);
        synthetic_max = synthetic_max.max(error);
        let changed = x.mapv(|v| -v);
        let proposed = head.proposal_metrics(&x, &changed)?;
        let reference = decoder.log_probs(&changed);
        let mut proposal_errors = Vec::new();
        for (row, metric) in proposed.iter().enumerate() {
            let (exact, comparison_error) = gam_mpd::acceptance::kl_logits(cpu.row(row), reference.row(row));
            let difference = (exact - metric.kl_estimate).abs();
            if !exact.is_finite() || !difference.is_finite() {
                return Err("nonfinite synthetic metric comparison".into());
            }
            synthetic_max = synthetic_max.max(difference);
            proposal_errors.push(json!({"CPU_KL":exact,"CPU_comparison_error":comparison_error,"proposal":metric,"absolute_difference":difference}));
        }
        synthetic.push(json!({"scale":scale,"max_log_probability_absolute_error":error,"proposal_metrics":proposal_errors}));
    }
    if head
        .log_probs(&ndarray::Array2::from_elem((1, width), f64::NAN))
        .is_ok()
    {
        return Err("nonfinite residual unexpectedly accepted".into());
    }
    // Explicit CUDA-kernel malformed-input checks, beyond the resident input guard.
    let finite = device.upload(ndarray::Array2::zeros((1, 2)).view()).map_err(|e| e.to_string())?;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let malformed = device.upload(ndarray::Array2::from_elem((1, 2), bad).view()).map_err(|e| e.to_string())?;
        if device.kl_proposal_rows(&malformed, &finite).is_ok() {
            return Err("GPU metric accepted nonfinite input".into());
        }
    }
    let overflow = device.upload(ndarray::arr2(&[[f64::MAX, -f64::MAX]]).view()).map_err(|e| e.to_string())?;
    if device.kl_proposal_rows(&overflow, &finite).is_ok() {
        return Err("GPU metric accepted overflowing finite shifts".into());
    }
    let resident_bytes = head.resident_bytes();
    let tile_rows = head.tile_rows();
    drop(head);
    let mut results = Vec::new();
    let mut passes = synthetic_max <= tolerance;
    for path in &a[5..] {
        let artifact = Artifact::from_bytes(
            &std::fs::read(path).map_err(|e| e.to_string())?,
            &native.declarations,
        )?;
        artifact.validate_coverage(&native)?;
        let measure = |gpu_head: bool| -> Result<_, String> {
            let runner = LanguageRun::new(&decoder, &native, &spec, &passages, 1)?
                .with_cuda(device.clone(), 2 << 30)?;
            let runner = if gpu_head {
                runner.with_cuda_readout(budget)?
            } else {
                runner
            };
            let start = std::time::Instant::now();
            let scores = runner.episodes(&artifact)?;
            let log_probs = runner.native_episode_log_probs(0, 0..spec.rows.min(8))?;
            let oracle_wall = start.elapsed().as_secs_f64();
            let oracle_timing = runner.timing();
            let proposal_start = std::time::Instant::now();
            let proposals = if gpu_head {
                Some(runner.gpu_metric_proposals(&artifact)?)
            } else {
                None
            };
            let proposal_wall = proposal_start.elapsed().as_secs_f64();
            Ok((
                RunMeasure::of(scores),
                log_probs,
                oracle_timing,
                oracle_wall,
                proposals,
                proposal_wall,
                runner.timing(),
            ))
        };
        let cpu_arm = if metric_only { None } else { Some(measure(false)?) };
        let (gpu, q, gpu_timing, gpu_wall, proposals, proposal_wall, after_proposal_timing) = measure(true)?;
        let (cpu, p, cpu_timing, cpu_wall) = if let Some((scores, probs, timing, wall, proposals, proposal_wall, final_timing)) = cpu_arm {
            drop((proposals, proposal_wall, final_timing));
            (scores, probs, Some(timing), Some(wall))
        } else {
            // These aliases only skip the already-validated CPU-head comparison;
            // proposal reductions below still compare against actual CPU metrics.
            (gpu.clone(), q.clone(), None, None)
        };
        let proposals = proposals.ok_or("missing opt-in GPU proposals")?;
        if proposals.len() != gpu.episodes.len() || p.iter().chain(q.iter()).any(|x| !x.is_finite()) {
            return Err("nonfinite sample or proposal count mismatch".into());
        }
        let mut proposal_max_error = 0.0_f64;
        let mut proposal_top1_equal = true;
        let mut proposal_comparisons = Vec::new();
        for (oracle, proposal) in gpu.episodes.iter().zip(&proposals) {
            if oracle.id != proposal.id || oracle.group != proposal.group || oracle.unheld != proposal.unheld {
                return Err("proposal episode identity/coverage changed".into());
            }
            let difference = (oracle.kl - proposal.kl_estimate).abs();
            if !oracle.kl.is_finite() || !difference.is_finite() || !proposal.conditional_reduction_error_estimate.is_finite() {
                return Err("nonfinite proposal comparison".into());
            }
            proposal_max_error = proposal_max_error.max(difference);
            proposal_top1_equal &= oracle.top1_agree == proposal.top1_agree;
            proposal_comparisons.push(json!({"id":oracle.id,"group":oracle.group,"CPU_metric_KL":oracle.kl,"GPU_proposal":proposal,"absolute_difference":difference}));
        }
        passes &= proposal_max_error <= tolerance && proposal_top1_equal;
        let sample_error = p
            .iter()
            .zip(q.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max);
        let mut max_kl = 0.0_f64;
        let mut max_effect = 0.0_f64;
        let mut identical_top1 = true;
        for (c, g) in cpu.episodes.iter().zip(&gpu.episodes) {
            if c.id != g.id || c.group != g.group || c.unheld != g.unheld {
                return Err("episode identity or intervention coverage changed".into());
            }
            max_kl = max_kl.max((c.kl - g.kl).abs());
            max_effect = max_effect.max((c.native_effect - g.native_effect).abs());
            identical_top1 &= c.top1_agree == g.top1_agree;
        }
        let verdicts = [0.001, 0.01, 0.1, 1.0, 10.0].map(|eps| {
            // The same existing group KL comparison band; no new error certificate.
            let classify = |m: &RunMeasure| {
                if m.groups.iter().all(|g| {
                    (if g.2 == 0.0 {
                        g.1
                    } else {
                        (g.1 + g.2).next_up()
                    }) <= eps
                }) {
                    "verified"
                } else if m.groups.iter().any(|g| {
                    (if g.2 == 0.0 {
                        g.1
                    } else {
                        (g.1 - g.2).next_down()
                    })
                    .max(0.0)
                        > eps
                }) {
                    "violated"
                } else {
                    "unresolved"
                }
            };
            let c = classify(&cpu);
            let g = classify(&gpu);
            passes &= c == g;
            json!({"epsilon":eps,"CPU_head":c,"CUDA_head":g})
        });
        passes &= cpu.episodes.len() == gpu.episodes.len()
            && identical_top1
            && max_kl <= tolerance
            && max_effect <= tolerance
            && sample_error <= tolerance;
        results.push(json!({"artifact":path,"episodes":cpu.episodes.len(),"max_KL_absolute_error":max_kl,"max_native_effect_absolute_error":max_effect,"native_episode0_sample_max_log_probability_error":sample_error,"identical_top1":identical_top1,"verdicts":verdicts,"CPU_head":if metric_only { None } else { Some(&cpu) },"CUDA_head":gpu,"CPU_head_timing":cpu_timing,"CUDA_head_timing":gpu_timing,"CPU_head_wall_seconds":cpu_wall,"CUDA_head_wall_seconds":gpu_wall,"GPU_metric_proposal_wall_seconds_warm_teacher":proposal_wall,"GPU_metric_proposal_max_absolute_error":proposal_max_error,"GPU_metric_proposal_identical_top1":proposal_top1_equal,"GPU_metric_proposal_comparisons":proposal_comparisons,"after_GPU_metric_proposal_timing":after_proposal_timing}));
    }
    let report = json!({"CPU_metric_numerical_scope":"Independent operational reference under existing assumed exp/log ULP model; Rust f64 transcendental accuracy is unspecified. Comparison bands are conditional, not certified transcendental/GEMM/network intervals.","passes":passes,"metric_proposals_only":metric_only,"CPU_head_comparison_skipped":metric_only,"prior_GPU_head_parity_provenance":"caller protocol must identify independent head parity","parity_tolerance":tolerance,"scope":if metric_only { "CUDA f64 native head; CPU metric oracle then GPU metric proposals with the same immutable teacher residual cache. Head parity relies on separately recorded prior evidence. Proposals never decide acceptance; timings are warm teacher cache, not end-to-end speedup or formal network certificates." } else { "CUDA f64 explained forwards in both arms; unchanged CPU teacher residual forwards; native output head differs. Fixed CPU-first order, warmed CUDA head. GPU metric estimates are proposal-only, never acceptance evidence; independent CPU metrics provide every verdict. Proposal timing reuses the CPU teacher cache. No matched speedup or formal full-network rounding certificate claimed." },"resident_numeric_bytes":resident_bytes,"workspace_numeric_budget":budget.workspace_bytes,"tile_rows":tile_rows,"budget_excludes":"allocator metadata, CUDA context and library-private GEMM workspace; input model/teacher residual storage belongs to runner","synthetic":synthetic,"results":results});
    std::fs::write(
        &a[3],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !passes {
        return Err("CPU/CUDA parity check failed; measured report saved".into());
    }
    Ok(())
}

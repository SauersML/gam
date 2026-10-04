//! Native-only finite MLP proposal bank; fitting does not certify Local or Run.
//! TRAIN_EXPORT OUT_DIR max_bank=N [layer=0 train=8 valid=2 context=512]
//! [m_ranks=1,2,4,8 k_ranks=1,2,4,8 supports=all]
//! Fit rows must be disjoint from the later frontier evaluation export.
use gam_linalg::decompose::svd;
use gam_mpd::acceptance::{CostCache, structural_cost, units};
use gam_mpd::artifact::Artifact;
use gam_mpd::counterfactual::Decoder;
use gam_mpd::import::import_language_model;
use gam_mpd::native_mlp::{Family, Feature, NativeRule, design, features, with_native_rule};
use gam_mpd::operator_program::{FamilyInputs, OperatorProgram};
use gam_mpd::run_check::{layer_nodes, split_sites};
use gam_solve::gaussian_reml_multi_penalty::GaussianRemlMultiPenaltyProblem;
use ndarray::{Array2, Axis, s};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, process::Command, time::Instant};

fn hash(path: &Path) -> Result<String, String> {
    for (exe, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        if let Ok(out) = Command::new(exe).args(args).arg(path).output() {
            if out.status.success() {
                let text = String::from_utf8(out.stdout).map_err(|e| e.to_string())?;
                let h = text.split_whitespace().next().ok_or("empty hash")?;
                if h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Ok(h.into());
                }
            }
        }
    }
    Err("SHA-256 tool required".into())
}
fn ranks(text: &str) -> Result<Vec<usize>, String> {
    let v: Vec<usize> = text
        .split(',')
        .map(|s| s.parse().map_err(|_| format!("invalid rank {s}")))
        .collect::<Result<_, _>>()?;
    if v.is_empty() || v.iter().any(|r| *r == 0) {
        return Err("positive independent rank lists required".into());
    }
    let mut sorted = v.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != v.len() {
        return Err("duplicate ranks".into());
    }
    Ok(v)
}
fn collect(
    program: &OperatorProgram,
    family: &FamilyInputs,
    groups: &[Vec<usize>],
    read: usize,
    write: usize,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    // Prefix execution keeps original place indices and skips the native head.
    let mut prefix = program.clone();
    prefix.nodes.truncate(write + 1);
    prefix.output = write;
    let (mut hs, mut ys) = (Vec::new(), Vec::new());
    for rows in groups {
        let trace = prefix
            .execute(&family.select(rows), false)
            .map_err(|e| e.to_string())?;
        hs.push(trace.values[read].clone());
        ys.push(trace.values[write].clone());
    }
    let concat = |a: &Vec<Array2<f64>>| {
        ndarray::concatenate(Axis(0), &a.iter().map(|v| v.view()).collect::<Vec<_>>())
            .map_err(|e| e.to_string())
    };
    Ok((concat(&hs)?, concat(&ys)?))
}
fn mse(actual: &Array2<f64>, predicted: &Array2<f64>) -> f64 {
    actual
        .iter()
        .zip(predicted)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / actual.len() as f64
}
fn main() -> Result<(), String> {
    let started = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("TRAIN_EXPORT OUT_DIR max_bank=N [layer/train/valid/context/m_ranks/k_ranks/supports=...]".into());
    }
    let mut opts = BTreeMap::new();
    for arg in &args[2..] {
        let (k, v) = arg.split_once('=').ok_or("key=value required")?;
        if opts.insert(k, v).is_some() {
            return Err(format!("duplicate {k}"));
        }
    }
    for key in opts.keys() {
        if ![
            "max_bank", "layer", "train", "valid", "context", "m_ranks", "k_ranks", "supports",
        ]
        .contains(key)
        {
            return Err(format!("unknown option {key}"));
        }
    }
    let number = |key: &str, default: usize| -> Result<usize, String> {
        opts.get(key).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("invalid {key}"))
        })
    };
    let budget = number("max_bank", 0)?;
    if budget == 0 {
        return Err("explicit positive max_bank required".into());
    }
    let (layer, train, valid, context) = (
        number("layer", 0)?,
        number("train", 8)?,
        number("valid", 2)?,
        number("context", 512)?,
    );
    if train == 0 || valid == 0 || context == 0 {
        return Err("positive train, valid and context required".into());
    }
    let ms = ranks(opts.get("m_ranks").copied().unwrap_or("1,2,4,8"))?;
    let ks = ranks(opts.get("k_ranks").copied().unwrap_or("1,2,4,8"))?;
    let supports: Vec<Option<f64>> = opts
        .get("supports")
        .copied()
        .unwrap_or("all")
        .split(',')
        .map(|v| {
            if v == "all" {
                Ok(None)
            } else {
                let t: f64 = v.parse().map_err(|_| "invalid support")?;
                if !t.is_finite() || t <= 0.0 {
                    Err("positive finite support threshold".into())
                } else {
                    Ok(Some(t))
                }
            }
        })
        .collect::<Result<_, String>>()?;
    if !supports.contains(&None) {
        return Err("supports must include all".into());
    }
    let declared = ms
        .len()
        .checked_mul(ks.len())
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(supports.len()))
        .and_then(|n| n.checked_add(1))
        .ok_or("bank size overflow")?;
    if declared > budget {
        return Err(format!(
            "declared bank {declared} (including native) exceeds max_bank {budget}"
        ));
    }
    let export = Path::new(&args[0]);
    let out = Path::new(&args[1]);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let imported = import_language_model(
        export,
        train.checked_add(valid).ok_or("sequence count overflow")?,
        context,
    )?;
    let native = split_sites(&imported.program)?;
    let decoder = Decoder::from_export(export)?;
    let layers = layer_nodes(&native, decoder.layers())?;
    let sites = layers.get(layer).ok_or("layer outside native decoder")?;
    let matrix = |suffix: &str| {
        native
            .operators
            .iter()
            .find(|o| o.name == format!("blocks.{layer}.{suffix}"))
            .map(|o| o.matrix())
            .ok_or(format!("missing native {suffix}"))
    };
    let input_svd = svd(matrix("c_fc")?.view(), false).map_err(|e| e.to_string())?;
    let output_svd = svd(matrix("down_proj")?.view(), false).map_err(|e| e.to_string())?;
    if *ms.iter().max().unwrap() > input_svd.vt.nrows()
        || *ks.iter().max().unwrap() > output_svd.u.ncols()
    {
        return Err("declared ranks exceed native matrix dimensions".into());
    }
    let input_resolved = input_svd
        .singular_values
        .iter()
        .filter(|&&v| v > input_svd.band)
        .count();
    let output_resolved = output_svd
        .singular_values
        .iter()
        .filter(|&&v| v > output_svd.band)
        .count();
    if *ms.iter().max().unwrap() > input_resolved || *ks.iter().max().unwrap() > output_resolved {
        return Err("declared ranks exceed numerically resolved native singular directions".into());
    }
    // Hash every exported tensor, not merely the token table or JSON paths.
    let mut source_hashes = BTreeMap::new();
    for entry in std::fs::read_dir(export).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "f64" || e == "json") {
            source_hashes.insert(
                path.file_name().unwrap().to_string_lossy().to_string(),
                hash(&path)?,
            );
        }
    }
    let groups = units(&imported.contract.family);
    if groups.len() != train + valid {
        return Err("imported sequence count changed".into());
    }
    let (h, y) = collect(
        &native,
        &imported.contract.family,
        &groups[..train],
        sites.normed,
        sites.mlp,
    )?;
    let (vh, vy) = collect(
        &native,
        &imported.contract.family,
        &groups[train..],
        sites.normed,
        sites.mlp,
    )?;
    let base = Artifact::native(&native)?;
    let native_path = out.join("native.artifact");
    std::fs::write(&native_path, base.f32_literals()?.to_bytes()?).map_err(|e| e.to_string())?;
    // Frontier supplies its reserved native candidate; do not duplicate that label.
    let mut bank = Vec::<Value>::new();
    let mut evidence = Vec::<Value>::new();
    let mut costs = CostCache::default();
    for &m in &ms {
        for &k in &ks {
            for family in [Family::Affine, Family::Gelu, Family::Products] {
                let fit_started = Instant::now();
                let reads = input_svd.vt.slice(s![..m, ..]).to_owned();
                let writes = output_svd.u.slice(s![.., ..k]).t().to_owned();
                let fs = features(m, family);
                let x = design(&h.dot(&reads.t()), &fs)?;
                let target = y.dot(&writes.t());
                let fitted = (|| -> Result<_, String> {
                    let mut penalty = Array2::zeros((x.ncols(), x.ncols()));
                    for c in 1..x.ncols() {
                        let mean = x.column(c).sum() / x.nrows() as f64;
                        let variance = x.column(c).iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / x.nrows() as f64;
                        if !variance.is_finite() || variance <= 0.0 {
                            return Err(format!("zero/nonfinite feature variance at {c}"));
                        }
                        penalty[[c, c]] = variance;
                    }
                    GaussianRemlMultiPenaltyProblem::new(x.view(), target.view(), &[penalty], 1)
                        .map_err(|e| e.to_string())?
                        .fit(None)
                        .map_err(|e| e.to_string())
                })();
                for (support_index, &support) in supports.iter().enumerate() {
                    let label = format!("L{layer}-m{m}-K{k}-{family:?}-support{support_index}");
                    let generated = (|| -> Result<Value, String> {
                        let fit = fitted.as_ref().map_err(Clone::clone)?;
                        // Declared support variants prune whole coefficient rows; no marginal feature search.
                        let keep: Vec<usize> = fs
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| {
                                support.is_none_or(|t| {
                                    fit.coefficients
                                        .row(i + 1)
                                        .iter()
                                        .map(|v| v * v)
                                        .sum::<f64>()
                                        .sqrt()
                                        >= t
                                })
                            })
                            .map(|(i, _)| i)
                            .collect();
                        let mut coeff = Array2::zeros((keep.len() + 1, k));
                        coeff.row_mut(0).assign(&fit.coefficients.row(0));
                        for (dst, &src) in keep.iter().enumerate() {
                            coeff
                                .row_mut(dst + 1)
                                .assign(&fit.coefficients.row(src + 1));
                        }
                        let selected: Vec<Feature> = keep.iter().map(|&i| fs[i].clone()).collect();
                        let proposal = NativeRule {
                            reads: reads.clone(),
                            writes: writes.clone(),
                            features: selected,
                            coefficients: coeff,
                        };
                        let artifact =
                            with_native_rule(&base, &label, sites, &proposal)?.f32_literals()?;
                        let bytes = artifact.to_bytes()?;
                        let decoded = Artifact::from_bytes(&bytes, &native.declarations)?;
                        let cost = structural_cost(&decoded, &mut costs)?;
                        let path = out.join(format!("{label}.artifact"));
                        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
                        bank.push(json!({"label":label,"artifact":path.canonicalize().map_err(|e|e.to_string())?}));
                        let train_prediction = design(&h.dot(&reads.t()), &proposal.features)?
                            .dot(&proposal.coefficients)
                            .dot(&writes);
                        let validation_prediction =
                            design(&vh.dot(&reads.t()), &proposal.features)?
                                .dot(&proposal.coefficients)
                                .dot(&writes);
                        Ok(
                            json!({"cost":cost,"features":proposal.features.len(),"train_proposal_mse":mse(&y,&train_prediction),"valid_proposal_mse":mse(&vy,&validation_prediction),"sha256":hash(&path)?,"certificate":format!("{:?}",fit.certificate),"reml_lambdas":fit.evaluation.lambdas.to_vec(),"coefficient_comparison_roundoff":fit.coefficients_roundoff}),
                        )
                    })();
                    evidence.push(json!({"label":label,"m":m,"K":k,"family":family,"support_threshold":support,"result":generated,"fit_seconds":fit_started.elapsed().as_secs_f64()}));
                }
            }
        }
    }
    let report = json!({"scope":"native-only finite proposal bank; run exact decoded Local/Run frontier separately; no acceptance claim","native_fallback":true,"train_export":export.canonicalize().map_err(|e|e.to_string())?,"source_sha256":source_hashes,"tokens_sha256":hash(&export.join("tokens.f64"))?,"export_sha256":hash(&export.join("export.json"))?,"fit_sequences":(0..train).collect::<Vec<_>>(),"validation_sequences":(train..train+valid).collect::<Vec<_>>(),"context":context,"layer":layer,"m_ranks":ms,"k_ranks":ks,"supports":supports,"declared_candidates_including_native":declared,"generated_candidates_including_frontier_native":bank.len()+1,"native_control_artifact":native_path,"generation_complete":bank.len()+1==declared,"max_bank":budget,"prior":"intercept unpenalized; feature-centered empirical variance diagonal Gaussian prior; REML-selected strength/shared response dispersion","coordinates":"native c_fc right singular directions; native down_proj left singular directions","svd":{"input_band":input_svd.band,"output_band":output_svd.band,"input_resolved_rank":input_resolved,"output_resolved_rank":output_resolved,"input_selected_singular_values":input_svd.singular_values.iter().take(*ms.iter().max().unwrap()).copied().collect::<Vec<_>>(),"output_selected_singular_values":output_svd.singular_values.iter().take(*ks.iter().max().unwrap()).copied().collect::<Vec<_>>()},"support_scope":"optional explicit proposal enumeration only; no acceptance or calibration threshold; default all features retained with REML shrinkage","proposal_mse_precision":"pre-serialization fit diagnostics, not decoded acceptance evidence","intervention_scope":"fixed benchmark unchanged; removed native hidden/active places remain unheld and predict no effect; no dropped episodes","candidates":evidence,"seconds":started.elapsed().as_secs_f64()});
    std::fs::write(
        out.join("BANK.json"),
        serde_json::to_vec_pretty(&bank).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        out.join("generation.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

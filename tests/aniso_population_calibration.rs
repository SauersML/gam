use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 300;
const SEED: u64 = 7;
const N_POPS: usize = 5;
const PC_DIM: usize = 5;
const CENTERS: usize = 12;
const DUCHON_ORDER: usize = 0;
const DUCHON_POWER: usize = 8;
const DUCHON_LENGTH: f64 = 1.0;
const SIGNIFICANCE_ALPHA: f64 = 0.05;

/// Writes the demo fixture CSV (response + 5 standardized PCs) and computes
/// the population labels used to group post-CTN z-scores. Native Rust:
/// the test must not depend on external Python interpreters or libraries
/// being installed in the cargo-test job.
fn write_demo_fixture(csv_path: &Path) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let normal = Normal::new(0.0, 1.0).expect("standard normal must be valid");

    // Draw the (N x PC_DIM) PC matrix in row-major order so that the first
    // PC for row 0 is the first variate consumed from the deterministic
    // RNG stream.  This keeps the generated dataset reproducible across
    // platforms regardless of how ndarray lays out columns.
    let mut pcs = vec![[0.0f64; PC_DIM]; N];
    for row in pcs.iter_mut() {
        for slot in row.iter_mut() {
            *slot = normal.sample(&mut rng);
        }
    }

    // Mirror the original numpy-defined ground truth: a smooth function of
    // PC1 (linear + centered quadratic) plus a smooth function of PC3
    // (linear + bounded tanh nonlinearity), then add Gaussian noise.
    let pgs_raw: Vec<f64> = pcs
        .iter()
        .map(|row| {
            let pc1 = row[0];
            let pc3 = row[2];
            let shift = 1.4 * pc1 + 0.6 * (pc1 * pc1 - 1.0) + 1.0 * pc3 + 0.4 * pc3.tanh();
            shift + 0.6 * normal.sample(&mut rng)
        })
        .collect();

    let mut writer = csv::Writer::from_path(csv_path).expect("open fixture csv for write");
    let mut header = vec!["pgs_raw".to_string()];
    for i in 0..PC_DIM {
        header.push(format!("pc{}_std", i + 1));
    }
    writer.write_record(&header).expect("write fixture header");
    let mut record = Vec::with_capacity(PC_DIM + 1);
    for (i, row) in pcs.iter().enumerate() {
        record.clear();
        record.push(format!("{:.6}", pgs_raw[i]));
        for &v in row.iter() {
            record.push(format!("{v:.6}"));
        }
        writer.write_record(&record).expect("write fixture row");
    }
    writer.flush().expect("flush fixture csv");

    assign_populations(&pcs)
}

/// Partitions the synthetic individuals into `N_POPS` deterministic
/// populations whose centers vary along the (PC1, PC3) plane that drives the
/// ground-truth response.  We use ordered quantile bins of the score
/// `pc1 + pc3` rather than k-means so the assignment is reproducible without
/// pulling in a clustering library, while still producing five non-empty,
/// monotonically-ordered groups whose covariate distributions overlap the
/// CTN-relevant geometry.  The test only relies on the labels to group the
/// post-CTN z-scores; CTN itself never sees them.
fn assign_populations(pcs: &[[f64; PC_DIM]]) -> Vec<usize> {
    let n = pcs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let sa = pcs[a][0] + pcs[a][2];
        let sb = pcs[b][0] + pcs[b][2];
        sa.partial_cmp(&sb).expect("PC scores must be finite")
    });

    let mut pop = vec![0usize; n];
    let chunk = n / N_POPS;
    let remainder = n % N_POPS;
    let mut start = 0usize;
    for k in 0..N_POPS {
        let size = chunk + if k < remainder { 1 } else { 0 };
        for &idx in &order[start..start + size] {
            pop[idx] = k;
        }
        start += size;
    }
    assert_eq!(start, n);
    pop
}

fn formula() -> String {
    format!(
        "pgs_raw ~ duchon(pc1_std, pc2_std, pc3_std, pc4_std, pc5_std, \
         centers={CENTERS}, order={DUCHON_ORDER}, power={DUCHON_POWER}, \
         length_scale={DUCHON_LENGTH})"
    )
}

fn gam_binary() -> PathBuf {
    option_env!("CARGO_BIN_EXE_gam")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/gam"))
}

fn run_command(mut command: Command, label: &str) {
    let output = command.output().unwrap_or_else(|err| {
        panic!("failed to run {label}: {err}");
    });
    if !output.status.success() {
        panic!(
            "{label} failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn run_fit_predict(workdir: &Path, tag: &str, scale_dimensions: bool, csv_in: &Path) -> Vec<f64> {
    let model = workdir.join(format!("model_{tag}.json"));
    let pred = workdir.join(format!("pred_{tag}.csv"));
    let mut fit = Command::new(gam_binary());
    fit.current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("fit")
        .arg("--transformation-normal");
    if scale_dimensions {
        fit.arg("--scale-dimensions");
    }
    fit.arg("--out").arg(&model).arg(csv_in).arg(formula());
    run_command(fit, &format!("fit {tag}"));

    let mut predict = Command::new(gam_binary());
    predict
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("predict")
        .arg(&model)
        .arg(csv_in)
        .arg("--out")
        .arg(&pred);
    run_command(predict, &format!("predict {tag}"));

    read_prediction_z(&pred)
}

fn read_prediction_z(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open prediction csv");
    let headers = reader.headers().expect("read prediction headers").clone();
    let z_col = ["z", "z_score", "transformed", "eta", "mean"]
        .iter()
        .find_map(|name| headers.iter().position(|header| header == *name))
        .unwrap_or_else(|| panic!("no z-like column in prediction headers: {headers:?}"));
    reader
        .records()
        .map(|record| {
            let record = record.expect("read prediction record");
            record[z_col]
                .parse::<f64>()
                .expect("prediction value must be finite f64")
        })
        .collect()
}

fn grouped_values(values: &[f64], pop: &[usize]) -> Vec<Vec<f64>> {
    let mut groups = vec![Vec::new(); N_POPS];
    for (&value, &group) in values.iter().zip(pop.iter()) {
        groups[group].push(value);
    }
    groups
}

fn anova_p_value(groups: &[Vec<f64>]) -> f64 {
    let n: usize = groups.iter().map(Vec::len).sum();
    assert!(
        n > groups.len(),
        "ANOVA requires residual degrees of freedom"
    );
    assert!(
        groups.iter().all(|group| !group.is_empty()),
        "all population groups must be non-empty"
    );

    let grand_mean = groups.iter().flat_map(|group| group.iter()).sum::<f64>() / n as f64;
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for group in groups {
        let group_mean = group.iter().sum::<f64>() / group.len() as f64;
        ss_between += group.len() as f64 * (group_mean - grand_mean).powi(2);
        ss_within += group
            .iter()
            .map(|value| (value - group_mean).powi(2))
            .sum::<f64>();
    }

    let df_between = (groups.len() - 1) as f64;
    let df_within = (n - groups.len()) as f64;
    if ss_within <= f64::EPSILON {
        return if ss_between <= f64::EPSILON { 1.0 } else { 0.0 };
    }
    let statistic = (ss_between / df_between) / (ss_within / df_within);
    let f_dist = FisherSnedecor::new(df_between, df_within).expect("valid F distribution");
    1.0 - f_dist.cdf(statistic)
}

fn brown_forsythe_p_value(groups: &[Vec<f64>]) -> f64 {
    let deviations: Vec<Vec<f64>> = groups
        .iter()
        .map(|group| {
            let center = median(group);
            group.iter().map(|value| (value - center).abs()).collect()
        })
        .collect();
    anova_p_value(&deviations)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn population_summary(groups: &[Vec<f64>]) -> String {
    groups
        .iter()
        .enumerate()
        .map(|(idx, group)| {
            let mean = group.iter().sum::<f64>() / group.len() as f64;
            let variance = group
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / group.len() as f64;
            format!(
                "pop{idx}: n={}, mean={mean:+.4}, variance={variance:.4}",
                group.len()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn assert_populations_not_significantly_different(tag: &str, z: &[f64], pop: &[usize]) {
    assert!(
        z.iter().all(|value| value.is_finite()),
        "{tag}: all predicted z values must be finite"
    );
    let overall_mean = z.iter().sum::<f64>() / z.len() as f64;
    let overall_variance = z
        .iter()
        .map(|value| (value - overall_mean).powi(2))
        .sum::<f64>()
        / z.len() as f64;
    assert!(
        overall_mean.abs() <= 0.15 && (0.5..=2.0).contains(&overall_variance),
        "{tag}: normalized z values must remain on the standard-normal scale; \
         overall mean={overall_mean:+.4}, variance={overall_variance:.4}"
    );

    let groups = grouped_values(z, pop);
    let mean_p = anova_p_value(&groups);
    let variance_p = brown_forsythe_p_value(&groups);
    let summary = population_summary(&groups);
    assert!(
        mean_p >= SIGNIFICANCE_ALPHA,
        "{tag}: population means are significantly different by one-way ANOVA \
         (p={mean_p:.6}, alpha={SIGNIFICANCE_ALPHA}); {summary}"
    );
    assert!(
        variance_p >= SIGNIFICANCE_ALPHA,
        "{tag}: population variances are significantly different by Brown-Forsythe Levene test \
         (p={variance_p:.6}, alpha={SIGNIFICANCE_ALPHA}); {summary}"
    );
}

fn assert_not_bit_identical(left_tag: &str, left: &[f64], right_tag: &str, right: &[f64]) {
    assert_eq!(
        left.len(),
        right.len(),
        "{left_tag} and {right_tag} predictions must have the same length"
    );
    let identical = left
        .iter()
        .zip(right.iter())
        .all(|(&a, &b)| a.to_bits() == b.to_bits());
    assert!(
        !identical,
        "{left_tag} and {right_tag} predictions were bit-for-bit identical"
    );
}

#[test]
fn aniso_demo_population_z_scores_are_equalized_for_iso_and_aniso() {
    assert!(file!().ends_with(".rs"));
    let dir = tempfile::tempdir().expect("create tempdir");
    let csv_path = dir.path().join("data.csv");
    let pop = write_demo_fixture(&csv_path);

    let z_iso = run_fit_predict(dir.path(), "iso", false, &csv_path);
    let z_aniso = run_fit_predict(dir.path(), "aniso", true, &csv_path);

    assert_populations_not_significantly_different("iso", &z_iso, &pop);
    assert_populations_not_significantly_different("aniso", &z_aniso, &pop);
    assert_not_bit_identical("iso", &z_iso, "aniso", &z_aniso);
}

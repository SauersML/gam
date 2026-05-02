use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use std::path::{Path, PathBuf};
use std::process::Command;

const N: usize = 300;
const SEED: u64 = 7;
const N_POPS: usize = 5;
const PC_DIM: usize = 5;
const CENTERS: usize = 12;
const DUCHON_ORDER: usize = 1;
const DUCHON_POWER: usize = 8;
const DUCHON_LENGTH: f64 = 1.0;
const SIGNIFICANCE_ALPHA: f64 = 0.05;

fn write_demo_fixture_with_python(csv_path: &Path, pop_path: &Path) {
    let script = format!(
        r#"
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

N = {N}
SEED = {SEED}
N_POPS = {N_POPS}
PC_DIM = {PC_DIM}

csv_path = Path(sys.argv[1])
pop_path = Path(sys.argv[2])

rng = np.random.default_rng(SEED)
pcs = rng.standard_normal((N, PC_DIM))
shift = (
    1.4 * pcs[:, 0]
    + 0.6 * (pcs[:, 0] ** 2 - 1.0)
    + 1.0 * pcs[:, 2]
    + 0.4 * np.tanh(pcs[:, 2])
)
pgs_raw = shift + 0.6 * rng.standard_normal(N)

coords = pcs[:, [0, 2]]
km = KMeans(n_clusters=N_POPS, n_init=10, random_state=SEED).fit(coords)
order = np.argsort(km.cluster_centers_[:, 0] + km.cluster_centers_[:, 1])
relabel = np.empty(N_POPS, dtype=int)
relabel[order] = np.arange(N_POPS)
pop = relabel[km.labels_]

with csv_path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["pgs_raw"] + ["pc%d_std" % (i + 1) for i in range(PC_DIM)])
    for i in range(N):
        writer.writerow(["%.6f" % pgs_raw[i]] + ["%.6f" % pcs[i, j] for j in range(PC_DIM)])

with pop_path.open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["pop"])
    for value in pop:
        writer.writerow([int(value)])
"#
    );
    let mut command = Command::new("python3");
    command.arg("-c").arg(script).arg(csv_path).arg(pop_path);
    run_command(command, "generate aniso demo fixture");
}

fn read_populations(path: &Path) -> Vec<usize> {
    let mut reader = csv::Reader::from_path(path).expect("open population csv");
    reader
        .records()
        .map(|record| {
            let record = record.expect("read population record");
            record[0]
                .parse::<usize>()
                .expect("population label must be usize")
        })
        .collect()
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
    if sorted.len() % 2 == 0 {
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
    let dir = tempfile::tempdir().expect("create tempdir");
    let csv_path = dir.path().join("data.csv");
    let pop_path = dir.path().join("pop.csv");
    write_demo_fixture_with_python(&csv_path, &pop_path);
    let pop = read_populations(&pop_path);

    let z_iso = run_fit_predict(dir.path(), "iso", false, &csv_path);
    let z_aniso = run_fit_predict(dir.path(), "aniso", true, &csv_path);

    assert_populations_not_significantly_different("iso", &z_iso, &pop);
    assert_populations_not_significantly_different("aniso", &z_aniso, &pop);
    assert_not_bit_identical("iso", &z_iso, "aniso", &z_aniso);
}

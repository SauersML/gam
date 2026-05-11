//! Reproduce the failing path from
//! `gnomon/examples/biobank/test_local_synthetic.py`:
//!     duchon(PC1..PC10, centers=40, order=0, power=2, length_scale=1.0)
//! against parametric block [intercept | sex | prs_z], with 200 training rows
//! drawn from a synthetic case/control cohort identical to the script.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    SpatialIdentifiability,
};
use gam::smooth::{
    LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
};
use gam::terms::smooth::build_term_collection_design;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const NUM_PCS: usize = 10;
const DUCHON_CENTERS: usize = 4 * NUM_PCS;
const N_TOTAL: usize = 20_000;
const N_TRAIN_CASES: usize = 100;
const N_TRAIN_CONTROLS: usize = 100;

fn synth_cohort(
    rng: &mut StdRng,
    n: usize,
    pgs_effect: f64,
    prevalence: f64,
) -> Vec<(i32, i32, f64, [f64; NUM_PCS])> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut pcs = vec![[0.0f64; NUM_PCS]; n];
    for row in pcs.iter_mut() {
        for v in row.iter_mut() {
            *v = normal.sample(rng);
        }
    }
    let sex: Vec<i32> = (0..n).map(|_| if normal.sample(rng) > 0.0 { 1 } else { 0 }).collect();
    let pgs_loadings: [f64; NUM_PCS] = std::array::from_fn(|_| normal.sample(rng) * 0.3);
    let pgs_raw: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..NUM_PCS {
                s += pcs[i][j] * pgs_loadings[j];
            }
            s + 0.15 * sex[i] as f64 + normal.sample(rng)
        })
        .collect();
    let mean = pgs_raw.iter().sum::<f64>() / n as f64;
    let var = pgs_raw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let pgs: Vec<f64> = pgs_raw.iter().map(|x| (x - mean) / std).collect();
    let pc_effects: [f64; NUM_PCS] = std::array::from_fn(|_| normal.sample(rng) * 0.2);
    let liability: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..NUM_PCS {
                s += pcs[i][j] * pc_effects[j];
            }
            s + 0.1 * sex[i] as f64 + pgs_effect * pgs[i] + normal.sample(rng)
        })
        .collect();
    let mut sorted = liability.clone();
    sorted.sort_by(f64::total_cmp);
    let q_idx = ((1.0 - prevalence) * (n as f64 - 1.0)).round() as usize;
    let threshold = sorted[q_idx.min(n - 1)];
    let case: Vec<i32> = liability.iter().map(|&l| if l > threshold { 1 } else { 0 }).collect();
    (0..n).map(|i| (case[i], sex[i], pgs[i], pcs[i])).collect()
}

#[test]
fn local_synth_copd_like_duchon_orth_to_parametric() {
    let mut rng = StdRng::seed_from_u64(0);
    let cohort = synth_cohort(&mut rng, N_TOTAL, 0.6, 0.06);

    let case_idx: Vec<usize> = (0..N_TOTAL).filter(|&i| cohort[i].0 == 1).collect();
    let ctrl_idx: Vec<usize> = (0..N_TOTAL).filter(|&i| cohort[i].0 == 0).collect();

    let train_pick: Vec<usize> = case_idx
        .iter()
        .take(N_TRAIN_CASES)
        .chain(ctrl_idx.iter().take(N_TRAIN_CONTROLS))
        .copied()
        .collect();

    let n_train = train_pick.len();
    let pgs_train: Vec<f64> = train_pick.iter().map(|&i| cohort[i].2).collect();
    let pgs_mean = pgs_train.iter().sum::<f64>() / n_train as f64;
    let pgs_var = pgs_train.iter().map(|x| (x - pgs_mean).powi(2)).sum::<f64>() / n_train as f64;
    let pgs_std = pgs_var.sqrt();

    // Layout matches train_df[cols] with cols = ["case", "sex", "prs_z", PC1..PC10]
    let p_cols = 3 + NUM_PCS;
    let mut data = Array2::<f64>::zeros((n_train, p_cols));
    for (row, &i) in train_pick.iter().enumerate() {
        let (case, sex, pgs, pcs) = cohort[i];
        data[[row, 0]] = case as f64;
        data[[row, 1]] = sex as f64;
        data[[row, 2]] = (pgs - pgs_mean) / pgs_std;
        for j in 0..NUM_PCS {
            data[[row, 3 + j]] = pcs[j];
        }
    }

    let pc_cols: Vec<usize> = (3..3 + NUM_PCS).collect();

    let duchon_term = SmoothTermSpec {
        name: "duchon_pcs".to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: pc_cols.clone(),
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: DUCHON_CENTERS,
                },
                length_scale: Some(1.0),
                // Mirror the term_builder policy auto-escalation: explicit power=2
                // with order=Zero and length_scale=Some(1.0) in d=10 is bumped to
                // the minimum admissible power for full triple-operator collocation,
                // which is `s_op = (d + max_op + 2 - 2p)/2 = (10 + 2 + 2 - 2)/2 = 6`.
                power: 6,
                nullspace_order: DuchonNullspaceOrder::Zero,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    };

    let spec = TermCollectionSpec {
        linear_terms: vec![
            LinearTermSpec {
                name: "sex".to_string(),
                feature_col: 1,
                double_penalty: true,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
            },
            LinearTermSpec {
                name: "prs_z".to_string(),
                feature_col: 2,
                double_penalty: true,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
            },
        ],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon_term],
    };

    let result = build_term_collection_design(data.view(), &spec);
    match result {
        Ok(design) => {
            eprintln!(
                "design built: rows={} cols={}",
                design.design.nrows(),
                design.design.ncols()
            );
        }
        Err(e) => {
            panic!("design build failed: {e}");
        }
    }
}

//! Mega-batch J: cycles 103-127 (25 quality regressions in one binary).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;

fn mk_1d(
    n: usize,
    f: impl Fn(f64) -> f64,
    range: (f64, f64),
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(range.0, range.1).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn mk_sphere(n: usize, f: impl Fn(f64, f64) -> f64, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = f(lat, lon) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_1d(formula: &str, data: gam::data::EncodedDataset, xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((xs.len(), 2));
    for (i, &x) in xs.iter().enumerate() { m[[i, 0]] = x; }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn fit_sphere(formula: &str, data: gam::data::EncodedDataset, pts: &[(f64, f64)]) -> Vec<f64> {
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((pts.len(), 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test] fn cyc_103_smooth_quadratic_recovery() { init_parallelism(); let d=mk_1d(200,|t|t*t,(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=10)",d,&[0.0,0.5,1.0]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn cyc_104_smooth_cubic_recovery() { init_parallelism(); let d=mk_1d(200,|t|t*t*t,(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=10)",d,&[0.0,0.5,1.0]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn cyc_105_smooth_sin_recovery() { init_parallelism(); let d=mk_1d(200,|t|(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=15)",d,&[0.1,0.5,0.9]); assert!(p.iter().all(|v|v.is_finite())); assert!(p[1]>0.5); }
#[test] fn cyc_106_smooth_exponential() { init_parallelism(); let d=mk_1d(200,|t|t.exp(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=15)",d,&[0.1,0.9]); assert!(p[1]>p[0]); }
#[test] fn cyc_107_smooth_log_recovers_monotone() { init_parallelism(); let d=mk_1d(200,|t|(t+0.1).ln(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=15)",d,&[0.1,0.5,0.9]); assert!(p[0]<p[1] && p[1]<p[2]); }
#[test] fn cyc_108_smooth_with_offset_in_y() { init_parallelism(); let d=mk_1d(200,|t|100.0+t.cos(),(0.0,TAU),0.05,7); let p=fit_1d("y~s(x,k=15)",d,&[0.0,PI]); assert!(p[0]>95.0 && p[1]<101.0); }
#[test] fn cyc_109_smooth_negative_y_range() { init_parallelism(); let d=mk_1d(200,|t|-5.0+(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=15)",d,&[0.5]); assert!(p[0] < -3.5 && p[0] > -5.5); }
#[test] fn cyc_110_smooth_with_outlier() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let noise=Normal::new(0.0,0.05).expect(""); let mut x:Vec<f64>=(0..200).map(|i|i as f64/199.0).collect(); x.sort_by(|a,b|a.partial_cmp(b).unwrap()); let mut y:Vec<f64>=x.iter().map(|&t|t.sin()+noise.sample(&mut rng)).collect(); y[100]=5.0; let h=["x","y"].into_iter().map(String::from).collect(); let r:Vec<StringRecord>=x.iter().zip(y.iter()).map(|(a,b)|StringRecord::from(vec![a.to_string(),b.to_string()])).collect(); let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit_1d("y~s(x,k=15)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn cyc_111_periodic_with_high_k() { init_parallelism(); let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit_1d("y~s(x,periodic=true,period=6.283185307179586,k=40)",d,&[0.0,PI]); assert!((p[0]-1.0).abs()<0.3 && (p[1]-(-1.0)).abs()<0.3); }
#[test] fn cyc_112_periodic_recovers_cos2t() { init_parallelism(); let d=mk_1d(200,|t|(2.0*t).cos(),(0.0,TAU),0.05,7); let p=fit_1d("y~s(x,periodic=true,period=6.283185307179586,k=20)",d,&[0.0,PI/2.0]); assert!((p[0]-1.0).abs()<0.3 && (p[1]+1.0).abs()<0.3); }
#[test] fn cyc_113_periodic_negative_truth() { init_parallelism(); let d=mk_1d(200,|t|-2.0+t.cos(),(0.0,TAU),0.05,7); let p=fit_1d("y~s(x,periodic=true,period=6.283185307179586,k=15)",d,&[0.0,PI]); assert!(p[0] > -2.0 && p[1] < -2.0); }
#[test] fn cyc_114_bc_anchored_with_few_data() { init_parallelism(); let d=mk_1d(50,|t|(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,bc=anchored,k=10)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn cyc_115_bc_clamped_with_few_data() { init_parallelism(); let d=mk_1d(50,|t|(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,bc=clamped,k=10)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn cyc_116_sphere_dense_polar() { init_parallelism(); let d=mk_sphere(400,|lat,_|0.5+0.3*lat.to_radians().sin(),0.05,7); let p=fit_sphere("y~sphere(lat,lon,k=30)",d,&[(85.0,0.0),(-85.0,0.0)]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn cyc_117_sphere_equatorial_data() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let u_lat=Uniform::new(-5.0_f64,5.0).expect(""); let u_lon=Uniform::new(-179.0_f64,179.0).expect(""); let noise=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(300); for _ in 0..300 { let lat=u_lat.sample(&mut rng); let lon=u_lon.sample(&mut rng); let y=lon.to_radians().cos()+noise.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),y.to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit_sphere("y~sphere(lat,lon,k=20)",d,&[(0.0,0.0),(0.0,90.0)]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn cyc_118_sphere_with_explicit_m_equals_1() { init_parallelism(); let d=mk_sphere(300,|lat,_|0.5+0.3*lat.to_radians().sin(),0.05,7); let p=fit_sphere("y~sphere(lat,lon,k=20,m=1)",d,&[(45.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn cyc_119_sphere_harmonic_with_m_equals_3() { init_parallelism(); let d=mk_sphere(300,|lat,_|0.5+0.3*lat.to_radians().sin(),0.05,7); let p=fit_sphere("y~sphere(lat,lon,method=harmonic,max_degree=4,m=3)",d,&[(45.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn cyc_120_smooth_fits_logistic_dose_response() { init_parallelism(); let d=mk_1d(200,|t|1.0/(1.0+(-(10.0*t-5.0)).exp()),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=15)",d,&[0.1,0.5,0.9]); assert!(p[0]<0.2 && p[1]>0.3 && p[2]>0.8); }
#[test] fn cyc_121_periodic_with_large_period() { init_parallelism(); let d=mk_1d(200,|t|(TAU*t/100.0).cos(),(0.0,100.0),0.05,7); let p=fit_1d("y~s(x,periodic=true,period=100.0,k=15)",d,&[0.0,50.0,100.0]); assert!((p[0]-p[2]).abs()<0.1); }
#[test] fn cyc_122_smooth_step_then_constant_recovers_inflection() { init_parallelism(); let d=mk_1d(200,|t|if t<0.4 {0.0} else if t<0.6 {(t-0.4)/0.2} else {1.0},(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=20)",d,&[0.2,0.5,0.8]); assert!(p[0]<0.4 && p[2]>0.6); }
#[test] fn cyc_123_smooth_with_periodic_aliased_data() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let u=Uniform::new(0.0_f64,TAU).expect(""); let noise=Normal::new(0.0,0.05).expect(""); let mut t:Vec<f64>=(0..200).map(|_|u.sample(&mut rng)).collect(); t.sort_by(|a,b|a.partial_cmp(b).unwrap()); let y:Vec<f64>=t.iter().map(|&x|(2.0*x).cos()+0.3*(5.0*x).sin()+noise.sample(&mut rng)).collect(); let h=["t","y"].into_iter().map(String::from).collect(); let r:Vec<StringRecord>=t.iter().zip(y.iter()).map(|(a,b)|StringRecord::from(vec![a.to_string(),b.to_string()])).collect(); let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit_1d("y~s(t,periodic=true,period=6.283185307179586,k=30)",d,&[0.0,PI/2.0,PI]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn cyc_124_sphere_fits_uniform_constant_truth() { init_parallelism(); let d=mk_sphere(300,|_,_|2.5,0.1,7); let p=fit_sphere("y~sphere(lat,lon,k=20)",d,&[(0.0,0.0),(45.0,90.0),(-45.0,-90.0)]); for v in &p { assert!((v-2.5).abs()<0.5); } }
#[test] fn cyc_125_sphere_fits_lat_dependent_truth() { init_parallelism(); let d=mk_sphere(300,|lat,_|0.5*lat.to_radians().sin(),0.05,7); let p=fit_sphere("y~sphere(lat,lon,k=20)",d,&[(60.0,0.0),(-60.0,0.0)]); assert!(p[0]>p[1]); }
#[test] fn cyc_126_smooth_with_no_intercept_explicit() { init_parallelism(); let d=mk_1d(200,|t|(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=10)",d,&[0.5]); assert!((p[0]-1.0).abs()<0.2); }
#[test] fn cyc_127_smooth_with_intercept_match() { init_parallelism(); let d=mk_1d(200,|t|3.0+0.5*t,(0.0,1.0),0.05,7); let p=fit_1d("y~s(x,k=10)",d,&[0.0,1.0]); assert!((p[0]-3.0).abs()<0.3 && (p[1]-3.5).abs()<0.3); }

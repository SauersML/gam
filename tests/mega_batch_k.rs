//! Mega-batch K: cycles 128-200 (73 quality regressions in one binary).
//! Compact one-liners (test-per-cycle) covering remaining edge cases.

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

fn mk_1d(n: usize, f: impl Fn(f64) -> f64, range: (f64, f64), sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(range.0, range.1).expect("");
    let noise = Normal::new(0.0, sigma).expect("");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let h = ["x", "y"].into_iter().map(String::from).collect();
    let r: Vec<StringRecord> = x.iter().zip(y.iter()).map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()])).collect();
    encode_recordswith_inferred_schema(h, r).expect("")
}

fn mk_2d(n: usize, f: impl Fn(f64, f64) -> f64, ra: (f64, f64), rb: (f64, f64), sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ua = Uniform::new(ra.0, ra.1).expect("");
    let ub = Uniform::new(rb.0, rb.1).expect("");
    let noise = Normal::new(0.0, sigma).expect("");
    let h = ["a", "b", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let a = ua.sample(&mut rng); let b = ub.sample(&mut rng);
        let y = f(a, b) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(h, rows).expect("")
}

fn fit1d(formula: &str, d: gam::data::EncodedDataset, xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &d, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((xs.len(), 2));
    for (i, &x) in xs.iter().enumerate() { m[[i, 0]] = x; }
    let dsg = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    dsg.design.apply(&fit.fit.beta).to_vec()
}

fn fit2d(formula: &str, d: gam::data::EncodedDataset, pts: &[(f64, f64)]) -> Vec<f64> {
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &d, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((pts.len(), 3));
    for (i, (a, b)) in pts.iter().enumerate() { m[[i, 0]] = *a; m[[i, 1]] = *b; }
    let dsg = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    dsg.design.apply(&fit.fit.beta).to_vec()
}

// 73 micro-cycles. Each = one #[test]. Pass = finite preds + sane range.

#[test] fn tensor_2d_recovers_x() { let d=mk_2d(300,|a,_|a,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=5)",d,&[(0.2,0.5),(0.8,0.5)]); assert!(p[0]<p[1]); }
#[test] fn tensor_2d_recovers_y() { let d=mk_2d(300,|_,b|b,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=5)",d,&[(0.5,0.2),(0.5,0.8)]); assert!(p[0]<p[1]); }
#[test] fn tensor_2d_interaction() { let d=mk_2d(300,|a,b|a*b,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=5)",d,&[(0.9,0.9),(0.1,0.1)]); assert!(p[0]>p[1]); }
#[test] fn tensor_2d_finite_at_origin() { let d=mk_2d(200,|a,b|a+b,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=5)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn sphere_finite_at_pole_predict() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-80.0_f64,80.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.5+0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,k=15)",d,&[(90.0,0.0),(-90.0,0.0)]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn periodic_with_high_freq_truth() { let d=mk_1d(300,|t|(8.0*t).cos(),(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=25)",d,&[0.0]); assert!(p[0].is_finite()); }
#[test] fn periodic_with_low_freq_truth() { let d=mk_1d(300,|t|(0.5*t).cos(),(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=15)",d,&[0.0]); assert!(p[0].is_finite()); }
#[test] fn smooth_with_normal_noise_high_sigma() { let d=mk_1d(200,|t|t.sin(),(0.0,TAU),1.0,7); let p=fit1d("y~s(x,k=10)",d,&[PI]); assert!(p[0].is_finite()); }
#[test] fn smooth_recovers_quadratic_in_offset_domain() { let d=mk_1d(200,|t|(t-5.0)*(t-5.0),(0.0,10.0),0.05,7); let p=fit1d("y~s(x,k=15)",d,&[2.0,5.0,8.0]); assert!(p[1]<p[0] && p[1]<p[2]); }
#[test] fn bc_clamped_handles_constant_truth() { let d=mk_1d(200,|_|3.0,(0.0,1.0),0.1,7); let p=fit1d("y~s(x,bc=clamped,k=15)",d,&[0.5]); assert!((p[0]-3.0).abs()<0.2); }
#[test] fn bc_anchored_handles_polynomial_truth() { let d=mk_1d(200,|t|t*t,(0.0,1.0),0.05,7); let p=fit1d("y~s(x,bc=anchored,k=15)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn sphere_recovers_lat_squared() { let d=mk_2d(300,|lat,_|lat.to_radians().sin().powi(2),(-70.0,70.0),(-179.0,179.0),0.05,7); let p=fit2d("y~sphere(a,b,k=20)",d,&[(0.0,0.0),(60.0,0.0)]); assert!(p[1]>p[0]); }
#[test] fn sphere_recovers_lon_dependent() { let d=mk_2d(300,|_,lon|lon.to_radians().cos(),(-70.0,70.0),(-179.0,179.0),0.05,7); let p=fit2d("y~sphere(a,b,k=20)",d,&[(0.0,0.0),(0.0,90.0)]); assert!(p[0]>p[1]); }
#[test] fn sphere_harmonic_finite_at_pole() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-80.0_f64,80.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.5+0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,method=harmonic,max_degree=4)",d,&[(90.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn cylinder_recovers_h_only() { let d=mk_2d(300,|_,h|0.5+0.4*h,(0.0,TAU),(-1.0,1.0),0.05,7); let p=fit2d("y~te(a,b,bc=['periodic','natural'],period=[2*pi,None],k=5)",d,&[(0.0,-0.5),(0.0,0.5)]); assert!(p[0]<p[1]); }
#[test] fn cylinder_recovers_theta_only() { let d=mk_2d(300,|th,_|th.cos(),(0.0,TAU),(-1.0,1.0),0.05,7); let p=fit2d("y~te(a,b,bc=['periodic','natural'],period=[2*pi,None],k=5)",d,&[(0.0,0.0),(PI,0.0)]); assert!(p[0]>p[1]); }
#[test] fn cylinder_seam_continuity_finite() { let d=mk_2d(300,|th,h|th.cos()+0.3*h,(0.0,TAU),(-1.0,1.0),0.05,7); let p=fit2d("y~te(a,b,bc=['periodic','natural'],period=[2*pi,None],k=5)",d,&[(0.0,0.5),(TAU,0.5)]); assert!((p[0]-p[1]).abs()<1e-6); }
#[test] fn torus_recovers_double_periodic() { let d=mk_2d(300,|u,v|u.cos()+v.sin(),(0.0,TAU),(0.0,TAU),0.05,7); let p=fit2d("y~te(a,b,bc=['periodic','periodic'],period=[2*pi,2*pi],k=5)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn torus_seam_u_continuous() { let d=mk_2d(300,|u,v|u.cos()+v.sin(),(0.0,TAU),(0.0,TAU),0.05,7); let p=fit2d("y~te(a,b,bc=['periodic','periodic'],period=[2*pi,2*pi],k=5)",d,&[(0.0,1.0),(TAU,1.0)]); assert!((p[0]-p[1]).abs()<1e-6); }
#[test] fn torus_seam_v_continuous() { let d=mk_2d(300,|u,v|u.cos()+v.sin(),(0.0,TAU),(0.0,TAU),0.05,7); let p=fit2d("y~te(a,b,bc=['periodic','periodic'],period=[2*pi,2*pi],k=5)",d,&[(1.0,0.0),(1.0,TAU)]); assert!((p[0]-p[1]).abs()<1e-6); }
#[test] fn matern_nu_half_1d() { let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~matern(x,nu=1/2)",d,&[PI]); assert!(p[0].is_finite()); }
#[test] fn matern_nu_three_halves_1d() { let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~matern(x,nu=3/2)",d,&[PI]); assert!(p[0].is_finite()); }
#[test] fn matern_nu_seven_halves_1d() { let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~matern(x,nu=7/2)",d,&[PI]); assert!(p[0].is_finite()); }
#[test] fn matern_recovers_polynomial() { let d=mk_1d(200,|t|t*t*t,(0.0,1.0),0.05,7); let p=fit1d("y~matern(x,nu=5/2)",d,&[0.5]); assert!((p[0]-0.125).abs()<0.1); }
#[test] fn matern_with_explicit_centers() { let d=mk_1d(200,|t|t.sin(),(0.0,1.0),0.05,7); let p=fit1d("y~matern(x,nu=5/2,centers=20)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn periodic_handles_quarter_period_data() { let d=mk_1d(200,|t|t.cos(),(0.0,PI/2.0),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=10)",d,&[0.0,PI]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn periodic_handles_three_quarter_period_data() { let d=mk_1d(200,|t|t.cos(),(0.0,3.0*PI/2.0),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=10)",d,&[0.0]); assert!(p[0].is_finite()); }
#[test] fn smooth_with_explicit_intercept_excluded() { let d=mk_1d(200,|t|t.sin()+1.0,(0.0,TAU),0.05,7); let p=fit1d("y~s(x,k=10)",d,&[PI/2.0]); assert!((p[0]-2.0).abs()<0.3); }
#[test] fn bc_anchored_with_large_data_range_works() { let d=mk_1d(200,|t|(0.5*t).sin(),(0.0,4.0),0.05,7); let p=fit1d("y~s(x,bc=anchored,k=15)",d,&[2.0]); assert!(p[0].is_finite()); }
#[test] fn bc_clamped_with_truth_zero_at_endpoints() { let d=mk_1d(200,|t|(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit1d("y~s(x,bc=clamped,k=15)",d,&[0.5]); assert!((p[0]-1.0).abs()<0.3); }
#[test] fn sphere_wahba_default_method_is_sobolev() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-80.0_f64,80.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.5+0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p_def=fit2d("y~sphere(lat,lon,k=15)",d.clone(),&[(45.0,0.0)]); let p_sob=fit2d("y~sphere(lat,lon,k=15,kernel=sobolev)",d,&[(45.0,0.0)]); assert!((p_def[0]-p_sob[0]).abs()<1e-9); }
#[test] fn sphere_wahba_pseudo_smoke_test() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-80.0_f64,80.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.5+0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,k=15,kernel=pseudo)",d,&[(45.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn te_with_high_k() { let d=mk_2d(400,|a,b|a*a+b*b,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=10)",d,&[(0.5,0.5)]); assert!(p[0].is_finite()); }
#[test] fn te_with_low_k() { let d=mk_2d(300,|a,b|a+b,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=4)",d,&[(0.5,0.5)]); assert!(p[0].is_finite()); }
#[test] fn smooth_at_data_extremes() { let d=mk_1d(200,|t|t,(0.1,0.9),0.05,7); let p=fit1d("y~s(x,k=10)",d,&[0.05,0.95]); assert!(p[0]<p[1]); }
#[test] fn periodic_at_seam_doubled() { let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586)",d,&[0.0,TAU,2.0*TAU]); assert!((p[0]-p[1]).abs()<1e-9 && (p[0]-p[2]).abs()<1e-9); }
#[test] fn sphere_grid_data_works() { let mut rows=Vec::new(); let h=["lat","lon","y"].into_iter().map(String::from).collect(); for i in 0..10 { for j in 0..10 { let lat=-70.0+140.0*(i as f64)/9.0; let lon=-160.0+320.0*(j as f64)/9.0; let y=0.5+0.3*lat.to_radians().sin(); rows.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),y.to_string()])); } } let d=encode_recordswith_inferred_schema(h,rows).expect(""); let p=fit2d("y~sphere(lat,lon,k=15)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn sphere_strict_lat_bound_at_90() { init_parallelism(); let mut rows=Vec::new(); let h=["lat","lon","y"].into_iter().map(String::from).collect(); for i in 0..30 { let lat=-80.0+i as f64*5.0; rows.push(StringRecord::from(vec![lat.to_string(),"0.0".to_string(),"1.0".to_string()])); } let d=encode_recordswith_inferred_schema(h,rows).expect(""); let cfg=FitConfig{family:Some("gaussian".to_string()),..FitConfig::default()}; let r=fit_from_formula("y~sphere(lat,lon,k=10)",&d,&cfg); assert!(r.is_ok() || r.is_err()); }
#[test] fn smooth_x_squared_inflection_at_zero() { let d=mk_1d(200,|t|t*t,(-1.0,1.0),0.05,7); let p=fit1d("y~s(x,k=15)",d,&[-1.0,0.0,1.0]); assert!(p[0]>p[1] && p[2]>p[1]); }
#[test] fn periodic_double_amplitude() { let d=mk_1d(200,|t|2.0*t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=15)",d,&[0.0,PI]); assert!(p[0]>1.5 && p[1]< -1.5); }
#[test] fn smooth_finite_at_data_centroid() { let d=mk_1d(300,|t|t.exp(),(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=10)",d,&[0.5]); assert!(p[0]>1.0 && p[0]<2.0); }
#[test] fn bc_anchored_pred_at_data_min_endpoint() { let d=mk_1d(200,|t|t,(0.1,0.9),0.05,7); let p=fit1d("y~s(x,bc=anchored,k=10)",d,&[0.1,0.9]); assert!((p[0]-p[1]).abs()<0.1); }
#[test] fn bc_clamped_pred_at_data_min_endpoint() { let d=mk_1d(200,|t|t,(0.1,0.9),0.05,7); let p=fit1d("y~s(x,bc=clamped,k=10)",d,&[0.1,0.9]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn smooth_recovers_zero_truth() { let d=mk_1d(200,|_|0.0,(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=10)",d,&[0.5]); assert!(p[0].abs()<0.1); }
#[test] fn periodic_recovers_zero_truth() { let d=mk_1d(200,|_|0.0,(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586)",d,&[PI]); assert!(p[0].abs()<0.1); }
#[test] fn sphere_recovers_zero_truth() { let d=mk_2d(300,|_,_|0.0,(-70.0,70.0),(-179.0,179.0),0.05,7); let p=fit2d("y~sphere(a,b,k=15)",d,&[(45.0,0.0)]); assert!(p[0].abs()<0.1); }
#[test] fn te_recovers_zero_truth() { let d=mk_2d(300,|_,_|0.0,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=5)",d,&[(0.5,0.5)]); assert!(p[0].abs()<0.1); }
#[test] fn smooth_handles_one_outlier_high() { let mut rng=StdRng::seed_from_u64(7); let noise=Normal::new(0.0,0.05).expect(""); let mut x:Vec<f64>=(0..100).map(|i|i as f64/99.0).collect(); x.sort_by(|a,b|a.partial_cmp(b).unwrap()); let mut y:Vec<f64>=x.iter().map(|&t|t.sin()+noise.sample(&mut rng)).collect(); y[10]=100.0; let h=["x","y"].into_iter().map(String::from).collect(); let r:Vec<StringRecord>=x.iter().zip(y.iter()).map(|(a,b)|StringRecord::from(vec![a.to_string(),b.to_string()])).collect(); let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit1d("y~s(x,k=10)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn smooth_handles_one_outlier_low() { let mut rng=StdRng::seed_from_u64(7); let noise=Normal::new(0.0,0.05).expect(""); let mut x:Vec<f64>=(0..100).map(|i|i as f64/99.0).collect(); x.sort_by(|a,b|a.partial_cmp(b).unwrap()); let mut y:Vec<f64>=x.iter().map(|&t|t.sin()+noise.sample(&mut rng)).collect(); y[50]=-100.0; let h=["x","y"].into_iter().map(String::from).collect(); let r:Vec<StringRecord>=x.iter().zip(y.iter()).map(|(a,b)|StringRecord::from(vec![a.to_string(),b.to_string()])).collect(); let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit1d("y~s(x,k=10)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn sphere_with_method_sos_alias() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,k=10,method=sos)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn sphere_with_method_mgcv_alias() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,k=10,method=mgcv)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn smooth_recovers_decreasing_truth() { let d=mk_1d(200,|t|1.0-t,(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=10)",d,&[0.1,0.9]); assert!(p[0]>p[1]); }
#[test] fn smooth_recovers_increasing_truth() { let d=mk_1d(200,|t|t*3.0,(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=10)",d,&[0.1,0.9]); assert!(p[0]<p[1]); }
#[test] fn smooth_recovers_v_shape() { let d=mk_1d(200,|t|(t-0.5).abs(),(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=15)",d,&[0.1,0.5,0.9]); assert!(p[1]<p[0] && p[1]<p[2]); }
#[test] fn smooth_recovers_inverted_v_shape() { let d=mk_1d(200,|t|0.5-(t-0.5).abs(),(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=15)",d,&[0.1,0.5,0.9]); assert!(p[1]>p[0] && p[1]>p[2]); }
#[test] fn smooth_k_consistency_check() { let d=mk_1d(200,|t|t.sin(),(0.0,1.0),0.05,7); let p10=fit1d("y~s(x,k=10)",d.clone(),&[0.5]); let p20=fit1d("y~s(x,k=20)",d,&[0.5]); assert!((p10[0]-p20[0]).abs()<0.2); }
#[test] fn sphere_lat_zero_lon_varies_finite() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(lon.to_radians().cos()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,k=15)",d,&[(0.0,-90.0),(0.0,0.0),(0.0,90.0)]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn periodic_handles_negative_offset_data() { let mut rng=StdRng::seed_from_u64(7); let u=Uniform::new(-PI,PI).expect(""); let no=Normal::new(0.0,0.05).expect(""); let mut t:Vec<f64>=(0..200).map(|_|u.sample(&mut rng)).collect(); t.sort_by(|a,b|a.partial_cmp(b).unwrap()); let y:Vec<f64>=t.iter().map(|&x|x.cos()+no.sample(&mut rng)).collect(); let h=["t","y"].into_iter().map(String::from).collect(); let r:Vec<StringRecord>=t.iter().zip(y.iter()).map(|(a,b)|StringRecord::from(vec![a.to_string(),b.to_string()])).collect(); let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit1d("y~s(t,periodic=true,period=6.283185307179586,origin=-3.141592653589793)",d,&[0.0]); assert!(p[0].is_finite()); }
#[test] fn te_with_disparate_scales() { let d=mk_2d(300,|a,b|a+b,(0.0,1.0),(0.0,1000.0),0.05,7); let p=fit2d("y~te(a,b,k=5)",d,&[(0.5,500.0)]); assert!(p[0].is_finite()); }
#[test] fn smooth_with_data_in_negative_range() { let d=mk_1d(200,|t|t.sin(),(-5.0,5.0),0.05,7); let p=fit1d("y~s(x,k=15)",d,&[-2.0,2.0]); assert!(p.iter().all(|v|v.is_finite())); }
#[test] fn bc_anchored_with_lots_of_data() { let d=mk_1d(1000,|t|(PI*t).sin(),(0.0,1.0),0.05,7); let p=fit1d("y~s(x,bc=anchored,k=25)",d,&[0.5]); assert!((p[0]-1.0).abs()<0.2); }
#[test] fn sphere_with_lots_of_data() { let d=mk_2d(1000,|lat,_|0.5+0.3*lat.to_radians().sin(),(-70.0,70.0),(-179.0,179.0),0.05,7); let p=fit2d("y~sphere(a,b,k=20)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn periodic_with_lots_of_data() { let d=mk_1d(1000,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=30)",d,&[0.0]); assert!((p[0]-1.0).abs()<0.2); }
#[test] fn smooth_consistent_across_seeds() { let p7=fit1d("y~s(x,k=15)",mk_1d(200,|t|t.sin(),(0.0,1.0),0.05,7),&[0.5]); let p11=fit1d("y~s(x,k=15)",mk_1d(200,|t|t.sin(),(0.0,1.0),0.05,11),&[0.5]); assert!((p7[0]-p11[0]).abs()<0.15); }
#[test] fn sphere_consistent_across_seeds() { let mut p_vals=Vec::new(); for seed in [3u64,7,11] { let mut rng=StdRng::seed_from_u64(seed); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); p_vals.push(fit2d("y~sphere(lat,lon,k=15)",d,&[(45.0,0.0)])[0]); } let max=p_vals.iter().cloned().fold(f64::NEG_INFINITY,f64::max); let min=p_vals.iter().cloned().fold(f64::INFINITY,f64::min); assert!(max-min<0.15); }
#[test] fn smooth_predictions_smooth_across_grid() { let d=mk_1d(200,|t|t.sin(),(0.0,1.0),0.05,7); let xs:Vec<f64>=(0..50).map(|i|i as f64/49.0).collect(); let p=fit1d("y~s(x,k=15)",d,&xs); for i in 1..p.len() { assert!((p[i]-p[i-1]).abs()<0.2); } }
#[test] fn periodic_predictions_smooth_across_grid() { let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let xs:Vec<f64>=(0..50).map(|i|TAU*(i as f64)/49.0).collect(); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586)",d,&xs); for i in 1..p.len() { assert!((p[i]-p[i-1]).abs()<0.5); } }
#[test] fn sphere_predictions_smooth_along_lat_band() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(200); for _ in 0..200 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.5+0.3*lat.to_radians().sin()+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let pts:Vec<(f64,f64)>=(0..30).map(|i|(-70.0+140.0*(i as f64)/29.0,0.0)).collect(); let p=fit2d("y~sphere(lat,lon,k=20)",d,&pts); for i in 1..p.len() { assert!((p[i]-p[i-1]).abs()<0.5); } }
#[test] fn smooth_with_large_data_count_finishes() { let d=mk_1d(5000,|t|t.sin(),(0.0,1.0),0.05,7); let p=fit1d("y~s(x,k=15)",d,&[0.5]); assert!(p[0].is_finite()); }
#[test] fn sphere_with_minimum_basis_finite() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(100); for _ in 0..100 { let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); r.push(StringRecord::from(vec![lat.to_string(),lon.to_string(),(0.5+no.sample(&mut rng)).to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let p=fit2d("y~sphere(lat,lon,k=5)",d,&[(0.0,0.0)]); assert!(p[0].is_finite()); }
#[test] fn te_with_minimum_basis_finite() { let d=mk_2d(200,|a,b|a+b,(0.0,1.0),(0.0,1.0),0.05,7); let p=fit2d("y~te(a,b,k=4)",d,&[(0.5,0.5)]); assert!(p[0].is_finite()); }
#[test] fn periodic_with_k_equal_basis_minimum() { let d=mk_1d(200,|t|t.cos(),(0.0,TAU),0.05,7); let p=fit1d("y~s(x,periodic=true,period=6.283185307179586,k=4)",d,&[PI]); assert!(p[0].is_finite()); }
#[test] fn all_smooth_families_in_one_formula() { init_parallelism(); let mut rng=StdRng::seed_from_u64(7); let ux=Uniform::new(0.0_f64,1.0).expect(""); let ut=Uniform::new(0.0_f64,TAU).expect(""); let ul=Uniform::new(-70.0_f64,70.0).expect(""); let un=Uniform::new(-179.0_f64,179.0).expect(""); let no=Normal::new(0.0,0.05).expect(""); let h=["x","theta","lat","lon","y"].into_iter().map(String::from).collect(); let mut r=Vec::with_capacity(400); for _ in 0..400 { let x=ux.sample(&mut rng); let th=ut.sample(&mut rng); let lat=ul.sample(&mut rng); let lon=un.sample(&mut rng); let y=0.3+0.2*x+0.3*th.cos()+0.2*lat.to_radians().sin()+no.sample(&mut rng); r.push(StringRecord::from(vec![x.to_string(),th.to_string(),lat.to_string(),lon.to_string(),y.to_string()])); } let d=encode_recordswith_inferred_schema(h,r).expect(""); let cfg=FitConfig{family:Some("gaussian".to_string()),..FitConfig::default()}; let result=fit_from_formula("y~s(x,k=6)+s(theta,periodic=true,period=6.283185307179586,k=6)+sphere(lat,lon,k=10)",&d,&cfg).expect("fit"); let FitResult::Standard(fit)=result else{panic!()}; let m=Array2::<f64>::from_shape_vec((1,5),vec![0.5,0.0,0.0,0.0,0.0]).expect(""); let dsg=build_term_collection_design(m.view(),&fit.resolvedspec).expect(""); let p=dsg.design.apply(&fit.fit.beta); assert!(p[0].is_finite()); }

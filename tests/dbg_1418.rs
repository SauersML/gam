//! TEMP debug — channel breakdown + frozen-θ̂ FD for owed_1418 fixtures.
use std::sync::Arc;
use ndarray::{Array1, Array2};
use gam::solver::arrow_schur::ArrowFactorCache;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm,
};

fn fixture(mode: AssignmentMode, log_lambda_sparse: f64) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 96usize; let p = 6usize; let k_atoms = 2usize; let m = 5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut logits = Array2::<f64>::zeros((n, k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    let dw = |b: usize, col: usize, atom: usize| 0.12 + 0.04*(b as f64) - 0.03*(col as f64) + 0.05*(atom as f64);
    for row in 0..n {
        let phase = (row as f64 + 0.5) / n as f64;
        coords[0][[row,0]] = phase; coords[1][[row,0]] = (phase+0.31).fract();
        let route = if row%2==0 {1.1} else {-1.1};
        logits[[row,0]] = route; logits[[row,1]] = if row%3==0 {0.8} else {0.4};
        let t0 = std::f64::consts::TAU*coords[0][[row,0]];
        let t1 = std::f64::consts::TAU*coords[1][[row,0]];
        let b0=[1.0,t0.sin(),t0.cos(),(2.0*t0).sin(),(2.0*t0).cos()];
        let b1=[1.0,t1.sin(),t1.cos(),(2.0*t1).sin(),(2.0*t1).cos()];
        let mix0 = 1.0/(1.0+(-route/0.7).exp()); let mix1 = 1.0-mix0;
        for col in 0..p {
            let mut v0=0.0; let mut v1=0.0;
            for b in 0..m { v0+=b0[b]*dw(b,col,0); v1+=b1[b]*dw(b,col,1); }
            let modelled = mix0*v0+mix1*v1;
            let high = 0.9*(5.0*t0+0.7*(col as f64)).sin()+0.6*(4.0*t1).cos()+0.25*(((row*7+col*13)%11) as f64 -5.0);
            target[[row,col]] = modelled+high;
        }
    }
    let mut atoms = Vec::new();
    for atom_idx in 0..k_atoms {
        let (phi,jet) = evaluator.evaluate(coords[atom_idx].view()).unwrap();
        let decoder = Array2::from_shape_fn((m,p),|(b,col)|dw(b,col,atom_idx));
        let mut smooth = Array2::<f64>::eye(m); smooth[[0,0]]=0.0;
        let atom = SaeManifoldAtom::new(format!("circle_{atom_idx}"),SaeAtomBasisKind::Periodic,1,phi,jet,decoder,smooth).unwrap()
            .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(logits,coords,vec![LatentManifold::Circle{period:1.0};k_atoms],mode).unwrap();
    let term = SaeManifoldTerm::new(atoms,assignment).unwrap();
    let rho = SaeManifoldRho::new(log_lambda_sparse,-2.0,vec![Array1::from_vec(vec![-1.0]),Array1::from_vec(vec![-1.0])]);
    (term,target,rho)
}

fn eval(start:&SaeManifoldTerm,target:&Array2<f64>,rho:&SaeManifoldRho,it:usize)->Option<(SaeManifoldTerm,f64,SaeManifoldLoss,ArrowFactorCache)>{
    let mut term=start.clone();
    match term.reml_criterion_with_cache(target.view(),rho,None,it,0.45,1.0e-6,1.0e-6) {
        Ok((v,l,c)) => Some((term,v,l,c)),
        Err(e) => { println!("  eval err (it={it}): {e}"); None }
    }
}

fn report(label:&str, mode: AssignmentMode) {
    println!("===== {label} =====");
    let (term,target,rho) = fixture(mode,-1.0);
    let it = 12usize;
    let Some((conv,_v,loss,cache)) = eval(&term,&target,&rho,it) else { println!("  base eval failed"); return; };
    println!("  base data_fit={:.4e}", loss.data_fit);
    let cmp = conv.analytic_outer_rho_gradient_at_converged(target.view(),&rho,&loss,&cache).unwrap();
    let g = cmp.gradient();
    let nparams = rho.to_flat().len();
    let h = 2.0e-4;
    for coord in 0..nparams {
        // re-solved FD (the owed gate's method)
        let mut fp = rho.to_flat(); let mut fm = rho.to_flat(); fp[coord]+=h; fm[coord]-=h;
        let resolved = match (eval(&conv,&target,&rho.from_flat(fp.view()),it), eval(&conv,&target,&rho.from_flat(fm.view()),it)) {
            (Some((_,vp,_,_)),Some((_,vm,_,_))) => Some((vp-vm)/(2.0*h)),
            _ => None,
        };
        // frozen-theta FD (inner_max_iter=0)
        let frozen = match (eval(&conv,&target,&rho.from_flat(fp.view()),0), eval(&conv,&target,&rho.from_flat(fm.view()),0)) {
            (Some((_,vp,lp,_)),Some((_,vm,lm,_))) => {
                let fl = (lp.total()-lm.total())/(2.0*h);
                let fr = ((vp-lp.total())-(vm-lm.total()))/(2.0*h);
                Some((fl,fr))
            }
            _ => None,
        };
        let an_expl = cmp.explicit[coord]; let an_rem = cmp.logdet_trace[coord]+cmp.occam[coord]; let an_third = cmp.third_order_correction[coord];
        println!("  coord{coord}: total_an={:.4e} resolved_fd={:?}  expl_an={:.4e} rem_an={:.4e} third_an={:.4e}  frozen_fd={:?}",
            g[coord], resolved.map(|x|format!("{x:.4e}")), an_expl, an_rem, an_third, frozen.map(|(a,b)|format!("expl={a:.4e} rem={b:.4e}")));
    }
}

fn deep_resolved(label:&str, mode: AssignmentMode) {
    println!("===== DEEP {label} (does re-solved FD coord0 -> analytic as iters grow?) =====");
    let (term,target,rho) = fixture(mode,-1.0);
    // Converge base deeply first.
    let Some((conv,_v,loss,cache)) = eval(&term,&target,&rho,200) else { println!("  base failed"); return; };
    let cmp = conv.analytic_outer_rho_gradient_at_converged(target.view(),&rho,&loss,&cache).unwrap();
    println!("  base(200) data_fit={:.4e} analytic_total[0]={:.4e} third[0]={:.4e}", loss.data_fit, cmp.gradient()[0], cmp.third_order_correction[0]);
    let h = 2.0e-4;
    for deep in [12usize, 50, 200, 600] {
        let mut fp = rho.to_flat(); let mut fm = rho.to_flat(); fp[0]+=h; fm[0]-=h;
        match (eval(&conv,&target,&rho.from_flat(fp.view()),deep), eval(&conv,&target,&rho.from_flat(fm.view()),deep)) {
            (Some((_,vp,lp,_)),Some((_,vm,lm,_))) => {
                let fd = (vp-vm)/(2.0*h);
                println!("  WARM deep={deep:>3}: resolved_fd[0]={:.4e}  df+={:.4e} df-={:.4e}", fd, lp.data_fit, lm.data_fit);
            }
            _ => println!("  WARM deep={deep:>3}: FD eval failed"),
        }
    }
    // COLD: re-solve from the ORIGINAL term (not the converged base) at each rho.
    for deep in [50usize, 200, 600] {
        let mut fp = rho.to_flat(); let mut fm = rho.to_flat(); fp[0]+=h; fm[0]-=h;
        match (eval(&term,&target,&rho.from_flat(fp.view()),deep), eval(&term,&target,&rho.from_flat(fm.view()),deep)) {
            (Some((_,vp,lp,_)),Some((_,vm,lm,_))) => {
                let fd = (vp-vm)/(2.0*h);
                println!("  COLD deep={deep:>3}: resolved_fd[0]={:.4e}  df+={:.4e} df-={:.4e}", fd, lp.data_fit, lm.data_fit);
            }
            _ => println!("  COLD deep={deep:>3}: FD eval failed"),
        }
    }
}

fn fixture_1417(mode: AssignmentMode, log_lambda_sparse: f64) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n=80usize; let p=6usize; let k_atoms=2usize; let m=5usize;
    let evaluator = PeriodicHarmonicEvaluator::new(m).unwrap();
    let mut logits = Array2::<f64>::zeros((n,k_atoms));
    let mut coords = vec![Array2::<f64>::zeros((n,1)),Array2::<f64>::zeros((n,1))];
    let mut target = Array2::<f64>::zeros((n,p));
    let w0=[[0.20,-0.10,0.06,0.03,-0.04,0.08],[0.70,-0.25,0.40,0.12,-0.35,0.18],[0.15,0.55,-0.25,0.28,0.08,-0.22],[0.08,-0.04,0.03,-0.02,0.01,0.06],[-0.06,0.02,0.05,0.04,-0.03,0.01]];
    let w1=[[-0.10,0.05,0.08,-0.02,0.05,-0.03],[-0.30,0.42,0.12,-0.20,0.16,0.30],[0.48,0.10,-0.32,0.18,0.26,-0.14],[0.04,0.07,-0.02,0.03,-0.05,0.02],[0.03,-0.05,0.04,0.01,0.02,-0.04]];
    for row in 0..n {
        let phase=(row as f64+0.25)/n as f64;
        coords[0][[row,0]]=phase; coords[1][[row,0]]=(phase+0.18).fract();
        let route = if row<n/2 {1.7} else {-1.7};
        logits[[row,0]]=route; logits[[row,1]]=if row%3==0 {0.9} else {0.3};
        let t0=std::f64::consts::TAU*coords[0][[row,0]]; let t1=std::f64::consts::TAU*coords[1][[row,0]];
        let b0=[1.0,t0.sin(),t0.cos(),(2.0*t0).sin(),(2.0*t0).cos()];
        let b1=[1.0,t1.sin(),t1.cos(),(2.0*t1).sin(),(2.0*t1).cos()];
        let mix0=1.0/(1.0+(-route/0.7).exp()); let mix1=1.0-mix0;
        for col in 0..p { let mut v0=0.0; let mut v1=0.0; for b in 0..m {v0+=b0[b]*w0[b][col]; v1+=b1[b]*w1[b][col];} target[[row,col]]=mix0*v0+mix1*v1; }
    }
    let mut atoms=Vec::new();
    for atom_idx in 0..k_atoms {
        let (phi,jet)=evaluator.evaluate(coords[atom_idx].view()).unwrap();
        let decoder=if atom_idx==0 {Array2::from_shape_fn((m,p),|(r,c)|w0[r][c])} else {Array2::from_shape_fn((m,p),|(r,c)|w1[r][c])};
        let mut smooth=Array2::<f64>::eye(m); smooth[[0,0]]=0.0;
        let atom=SaeManifoldAtom::new(format!("circle_{atom_idx}"),SaeAtomBasisKind::Periodic,1,phi,jet,decoder,smooth).unwrap()
            .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap()) as Arc<dyn SaeBasisEvaluator>);
        atoms.push(atom);
    }
    let assignment=SaeAssignment::from_blocks_with_mode_and_manifolds(logits,coords,vec![LatentManifold::Circle{period:1.0};k_atoms],mode).unwrap();
    let term=SaeManifoldTerm::new(atoms,assignment).unwrap();
    let rho=SaeManifoldRho::new(log_lambda_sparse,-8.0,vec![Array1::from_vec(vec![-8.0]),Array1::from_vec(vec![-8.0])]);
    (term,target,rho)
}

fn report_1417(label:&str, mode: AssignmentMode, lls: f64) {
    println!("===== 1417 {label} (lls={lls}) =====");
    let (term,target,rho)=fixture_1417(mode,lls);
    let Some((conv,_v,loss,cache))=eval(&term,&target,&rho,8) else { println!("  base eval failed"); return; };
    println!("  base data_fit={:.4e}", loss.data_fit);
    let cmp=conv.analytic_outer_rho_gradient_at_converged(target.view(),&rho,&loss,&cache).unwrap();
    let g=cmp.gradient(); let h=2.0e-4;
    for coord in 0..rho.to_flat().len() {
        let mut fp=rho.to_flat(); let mut fm=rho.to_flat(); fp[coord]+=h; fm[coord]-=h;
        let resolved=match (eval(&conv,&target,&rho.from_flat(fp.view()),8),eval(&conv,&target,&rho.from_flat(fm.view()),8)) {
            (Some((_,vp,_,_)),Some((_,vm,_,_)))=>Some((vp-vm)/(2.0*h)), _=>None };
        let frozen=match (eval(&conv,&target,&rho.from_flat(fp.view()),0),eval(&conv,&target,&rho.from_flat(fm.view()),0)) {
            (Some((_,vp,lp,_)),Some((_,vm,lm,_)))=>{ let fl=(lp.total()-lm.total())/(2.0*h); let fr=((vp-lp.total())-(vm-lm.total()))/(2.0*h); Some((fl,fr)) } _=>None };
        println!("  coord{coord}: tot_an={:.4e} resolved={:?} expl_an={:.4e} rem_an={:.4e} third_an={:.4e} frozen={:?}",
            g[coord], resolved.map(|x|format!("{x:.3e}")), cmp.explicit[coord], cmp.logdet_trace[coord]+cmp.occam[coord], cmp.third_order_correction[coord],
            frozen.map(|(a,b)|format!("expl={a:.3e} rem={b:.3e}")));
    }
}

#[test]
fn dbg_1417_breakdown() {
    report_1417("learnable_alpha", AssignmentMode::ibp_map(0.7,0.9,true), -1.5);
    report_1417("fixed_alpha", AssignmentMode::ibp_map(0.7,0.9,false), -1.5);
    assert!(true);
}

#[test]
fn dbg_1417_iter_sweep() {
    for (label,mode) in [("fixed",AssignmentMode::ibp_map(0.7,0.9,false)),("learn",AssignmentMode::ibp_map(0.7,0.9,true))] {
        println!("===== iter sweep {label} =====");
        // Sweep ρ_sparse: is there ANY prior strength where learnable-α is PD?
        for lls in [-4.0f64,-3.0,-2.0,-1.0,0.0,1.0,2.0] {
            let (t2,tg2,r2)=fixture_1417(mode,lls);
            match eval(&t2,&tg2,&r2,60) {
                Some((_,_,l,_)) => println!("  lls={lls:>5}: OK data_fit={:.4e}", l.data_fit),
                None => println!("  lls={lls:>5}: FAILED"),
            }
        }
    }
    assert!(true);
}

#[test]
fn dbg_1418_breakdown() {
    report("softmax", AssignmentMode::softmax(0.7));
    report("jumprelu", AssignmentMode::jumprelu(0.7, 0.0));
    report("ibp_learnable", AssignmentMode::ibp_map(0.7, 0.9, true));
    report("ibp_fixed", AssignmentMode::ibp_map(0.7, 0.9, false));
    deep_resolved("softmax", AssignmentMode::softmax(0.7));
    assert!(true);
}

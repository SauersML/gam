//! Explicit CUDA technical control; synthetic shared nonlinear fixture, not a method result.
use gam_mpd::{operator_program::{Declarations, Interface, Law, Node, Operator, OperatorProgram, Rule, Slot, SlotValues, FamilyInputs, exact_precision}, artifact_device::mapped_inlined, device_program::DeviceProgram, resident_rule_fit::{fit, Settings, ProposalArithmetic}};
use gam_gpu::{GpuPolicy, tensor::{Device, Arithmetic}};
use ndarray::{Array2, array};
use std::{sync::Arc, collections::BTreeMap, time::Instant};
fn model(weight: f64, offset: f64) -> OperatorProgram {
        let i = Interface::native(1).expect("interface");
        let c = Interface::constant();
        let op = |name: &str, value: f64, cols: &Interface| {
            Arc::new(
                Operator::dense(
                    name,
                    i.clone(),
                    cols.clone(),
                    Array2::from_elem((1, 1), value),
                    exact_precision([value]).expect("precision"),
                    Default::default(),
                )
                .expect("operator"),
            )
        };
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![
                op("shared weight", weight, &i),
                op("shared offset", offset, &c),
                op("second binding", 1.5, &i),
            ],
            rules: vec![Rule {
                name: "shared nonlinear function".into(),
                inputs: vec![i],
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Affine {
                        terms: vec![(0, 0)],
                        bias: Some(1),
                    },
                    Node::Pointwise {
                        input: 1,
                        laws: vec![Law::GeluTanh],
                    },
                ],
                output: 2,
            }],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 2)],
                    bias: None,
                },
                Node::Call {
                    rule: 0,
                    arguments: vec![0],
                },
                Node::Call {
                    rule: 0,
                    arguments: vec![1],
                },
                Node::Concat { parts: vec![2, 3] },
            ],
            output: 4,
        }
    }
fn target(p: &OperatorProgram, x: &Array2<f64>) -> Array2<f64> {
        p.execute(
            &FamilyInputs {
                rows: x.nrows(),
                slots: vec![SlotValues::Raw(x.clone())],
                layout: None,
            },
            false,
        )
        .expect("teacher")
        .values[p.output]
            .clone()
    }

fn difference(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64,String> {
    if a.dim()!=b.dim() { return Err("shape mismatch".into()); }
    a.iter().zip(b.iter()).try_fold(0.0_f64, |m, (&x,&y)| {
        if !x.is_finite() || !y.is_finite() { return Err("nonfinite parity value".into()); }
        Ok(m.max((x-y).abs()))
    })
}
fn run() -> Result<(),String> {
    let device=Device::accelerator(GpuPolicy::Required)?.ok_or("required accelerator absent")?;
    if !cfg!(target_os = "linux") || device.is_host() || !device.float64() { return Err(format!("required CUDA f64, got {}",device.name())); }
    let mut p=model(0.6,0.1);
    p.rules[0].nodes.push(Node::Gain{input:2,coefficient:gam_mpd::operator_program::Coefficient::Number(-0.75)});
    p.rules[0].output=3;
    let mut teacher=model(1.2,-0.2); teacher.rules=p.rules.clone();
    let x=array![[-1.2],[-0.3],[0.4],[1.1],[1.8]]; let vx=array![[-0.7],[0.8],[1.5]];
    let y=target(&teacher,&x); let vy=target(&teacher,&vx);
    let (expanded,_)=mapped_inlined(&p)?;
    let family=FamilyInputs{rows:x.nrows(),slots:vec![SlotValues::Raw(x.clone())],layout:None};
    let cpu=expanded.execute(&family,false).map_err(|e|e.to_string())?;
    let mut gradients=Vec::new(); let mut forward_error=0.0_f64;
    for backend in [Device::host(),device.clone()] {
        let mut lowered=DeviceProgram::compile_values(&backend,&expanded)?;
        lowered.prepare_dense_parameters(&[0,1])?;
        let trace=lowered.forward(&family)?;
        forward_error=forward_error.max(difference(&backend.download(trace.value(expanded.output)?)?,&cpu.values[expanded.output])?);
        let seed=Array2::from_elem((x.nrows(),2),0.37);
        let (_,g)=lowered.vjp_values_dense(&trace,BTreeMap::from([(expanded.output,backend.upload(seed.view())?)]),&[],&[0,1],Arithmetic::F64)?;
        gradients.push([backend.download(&g[&0])?,backend.download(&g[&1])?]);
    }
    let gradient_error=difference(&gradients[0][0],&gradients[1][0])?.max(difference(&gradients[0][1],&gradients[1][1])?);
    let mut finite_difference_error=0.0_f64;
    for op in [0,1] {
        let mut objectives=Vec::new();
        for direction in [-1.0,1.0] {
            let mut shifted=expanded.clone();
            match &mut Arc::make_mut(&mut shifted.operators[op]).body {
                gam_mpd::operator_program::OperatorBody::Dense{values,..} => values[[0,0]]+=direction*1e-5,
                _ => return Err("fixture parameter must be dense".into()),
            }
            let values=shifted.execute(&family,false).map_err(|e|e.to_string())?;
            objectives.push(values.values[shifted.output].sum()*0.37);
        }
        finite_difference_error=finite_difference_error.max(((objectives[1]-objectives[0])/2e-5-gradients[1][op][[0,0]]).abs());
    }
    if finite_difference_error>2e-8 { return Err(format!("independent CPU finite difference mismatch {finite_difference_error}")); }
    if forward_error>2e-12 || gradient_error>2e-12 { return Err(format!("parity failure forward={forward_error} gradient={gradient_error}")); }
    let mut fits=Vec::new();
    for (backend,arithmetic) in [(Device::host(),ProposalArithmetic::F64),(device.clone(),ProposalArithmetic::F64),(device.clone(),ProposalArithmetic::F32)] {
        let settings=Settings{iterations:32,forward_rows:3,learning_rate:0.02,beta1:0.9,beta2:0.99,epsilon:1e-8,numeric_bytes:16<<20,arithmetic};
        let start=Instant::now(); let fitted=fit(&backend,&p,&x,&y,&vx,&vy,&[0,1],settings)?;
        fits.push(serde_json::json!({"backend":backend.name(),"wall_seconds":start.elapsed().as_secs_f64(),"report":fitted.report}));
    }
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"scope":"synthetic technical control: shared full nonlinear activation, constant bias and fixed Gain; not native-data fit or acceptance", "backend":device.name(),"forward_max_abs":forward_error,"dense_gradient_max_abs":gradient_error,"cpu_finite_difference_max_abs":finite_difference_error,"fits":fits})).map_err(|e|e.to_string())?);
    Ok(())
}
fn main() -> Result<(),String> { run() }

//! Explicit SYNTHETIC calibration data for the existing composed-rule search driver.
use gam_mpd::operator_program::exact_precision;
use gam_mpd::{
    artifact::Artifact,
    engine::sha256,
    composed_rule_search::{self, Binary, Expr, Grammar, Unary, UseSpec},
    operator_program::{FamilyInputs, Operator, SlotValues},
};
use ndarray::{Array2, s};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        2. * ((self.next() >> 11) as f64 / 9007199254740992.) - 1.
    }
}
fn operations(e: &Expr) -> usize {
    match e {
        Expr::Argument(_) => 0,
        Expr::Unary(_, x) | Expr::Affine(x) => 1 + operations(x),
        Expr::Binary(_, a, b) => 1 + operations(a) + operations(b),
    }
}
fn eligible(e: &Expr) -> bool {
    fn internal_affine(e: &Expr) -> bool {
        matches!(e, Expr::Affine(x) if matches!(x.as_ref(), Expr::Unary(Unary::Relu, _)))
    }
    match e {
        Expr::Unary(Unary::Relu, x) => internal_affine(x) || eligible(x),
        Expr::Binary(Binary::Multiply, a, b) => {
            internal_affine(a) || internal_affine(b) || eligible(a) || eligible(b)
        }
        Expr::Unary(_, x) | Expr::Affine(x) => eligible(x),
        Expr::Binary(_, a, b) => eligible(a) || eligible(b),
        Expr::Argument(_) => false,
    }
}
fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn array(
    root: &Path,
    name: &str,
    values: &Array2<f64>,
    layer: usize,
    role: &str,
) -> Result<Value, String> {
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite synthetic teacher value".into());
    }
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let path = root.join(name);
    std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
    Ok(
        json!({"file":name,"sha256":sha256(&path)?,"width":values.ncols(),"layer":layer,"role":role,"encoding":"f64_le"}),
    )
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: mpd_composed_rule_calibration_2951 OUT_DIR".into());
    }
    let out = Path::new(&args[1]);
    if out.exists() {
        return Err("fresh OUT_DIR required".into());
    }
    let grammar = Grammar {
        arguments: 1,
        max_operations: 3,
        max_expressions: 10000,
        unary: vec![Unary::Relu],
        binary: vec![Binary::Multiply],
        affine: true,
    };
    let inventory = composed_rule_search::enumerate(&grammar)?;
    if inventory.truncated {
        return Err("calibration grammar must be completely enumerated".into());
    }
    let selection_seed = 29510001;
    let mut random = Random(selection_seed);
    let mut candidates: Vec<_> = inventory
        .expressions
        .iter()
        .enumerate()
        .filter(|(_, e)| operations(e) >= 3 && eligible(e))
        .collect();
    if candidates.is_empty() {
        return Err("no internal-affine nonlinear calibration targets".into());
    }
    let mut selected = Vec::new();
    while !candidates.is_empty() && selected.len() < 3 {
        let choice = (random.next() % candidates.len() as u64) as usize;
        let (index, expression) = candidates.remove(choice);
        selected.push((index, expression.clone()));
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut paths = Vec::new();
    for (dataset, (expression_id, expression)) in selected.into_iter().enumerate() {
        let root = out.join(format!("target{dataset}"));
        std::fs::create_dir(&root).map_err(|e| e.to_string())?;
        let teacher_seed = 29511000 + dataset as u64;
        let search_seed = 29512000 + dataset as u64;
        let mut proposal = composed_rule_search::compile(
            &expression,
            8,
            &[UseSpec {
                input_width: 8,
                output_width: 8,
            }; 2],
            teacher_seed,
        )?;
        let internal_seed = 29515000 + dataset as u64;
        let mut internal = Random(internal_seed);
        for op in &mut proposal.program.operators {
            let amplitude = match op.name.as_str() {
                "shared internal affine matrix" => (3.0_f64 / 8.0).sqrt(),
                "shared internal affine offset" => 0.2,
                _ => continue,
            };
            let values = Array2::from_shape_fn((op.rows.width(), op.cols.width()), |_| {
                amplitude * internal.unit()
            });
            let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
            *op = Arc::new(
                Operator::dense(
                    op.name.clone(),
                    op.rows.clone(),
                    op.cols.clone(),
                    values,
                    precision,
                    Default::default(),
                )
                .map_err(|e| e.to_string())?,
            );
        }
        let teacher = Artifact::native(&proposal.program)?.f32_literals()?;
        let bytes = teacher.to_bytes()?;
        let teacher_path = root.join("teacher.artifact");
        std::fs::write(&teacher_path, &bytes).map_err(|e| e.to_string())?;
        let decoded = Artifact::from_bytes(&bytes, &teacher.program.declarations)?;
        if decoded.to_bytes()? != bytes {
            return Err("teacher canonical wire roundtrip mismatch".into());
        }
        let teacher_sha = sha256(&teacher_path)?;
        let mut panels = Vec::new();
        for (name, rows, radius, seed) in [
            ("train", 128, 1., 29513000 + dataset as u64),
            ("eval", 256, 2., 29514000 + dataset as u64),
        ] {
            let mut rng = Random(seed);
            let inputs: Vec<_> = (0..2)
                .map(|_| Array2::from_shape_fn((rows, 8), |_| radius * rng.unit()))
                .collect();
            let family = FamilyInputs {
                rows,
                slots: inputs.iter().cloned().map(SlotValues::Raw).collect(),
                layout: None,
            };
            let trace = decoded
                .program
                .execute(&family, false)
                .map_err(|e| e.to_string())?;
            let output = &trace.values[decoded.program.output];
            let mut values = Vec::new();
            for (layer, input) in inputs.iter().enumerate() {
                values.push(array(
                    &root,
                    &format!("{name}-use{layer}-input.f64"),
                    input,
                    layer,
                    "input",
                )?);
                let write = output.slice(s![.., layer * 8..(layer + 1) * 8]).to_owned();
                values.push(array(
                    &root,
                    &format!("{name}-use{layer}-write.f64"),
                    &write,
                    layer,
                    "write",
                )?);
            }
            panels.push(json!({"name":name,"rows":rows,"record":{"scope":"SYNTHETIC CALIBRATION ONLY","source":{"checkpoint_sha256":teacher_sha},"config":{"synthetic":true,"width":8,"uses":2,"serialization":"f32"},"distribution":"uniform","range":[-radius,radius]},"values":values}));
        }
        save(
            &root.join("EXTRACT.json"),
            &json!({"scope":"SYNTHETIC CALIBRATION ONLY; no pretrained-model or blind discovery evidence","panels":panels}),
        )?;
        save(
            &root.join("SETTINGS.json"),
            &json!({"layers":[0,1],"width":8,"grammar":grammar,"seed":search_seed,"fit":{"iterations":128,"forward_rows":128,"learning_rate":0.01,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"numeric_bytes":536870912,"arithmetic":"f64","backtracking":null}}),
        )?;
        save(
            &root.join("GROUND_TRUTH.json"),
            &json!({"scope":"SYNTHETIC CALIBRATION ONLY; targets deliberately selected from search grammar","expression":expression,"inventory_expression_id":expression_id,"selection_seed":selection_seed,"selection":"PRNG sampling without replacement from eligible internal Affine after ReLU and before product or ReLU; at most three distinct targets","internal_seed":internal_seed,"internal_matrix_distribution":"zero-centered uniform[-sqrt(3/8),sqrt(3/8)]","internal_offset_distribution":"uniform[-0.2,0.2]","train_seed":29513000+dataset as u64,"eval_seed":29514000+dataset as u64,"teacher_seed":teacher_seed,"search_seed":search_seed,"teacher_sha256":sha256(&teacher_path)?,"teacher_execution":"ordinary Rust execution of f32 serialized decoded saved teacher","width":8,"uses":2,"maps":"full-width"}),
        )?;
        paths.push(
            json!({"extract":root.join("EXTRACT.json"),"settings":root.join("SETTINGS.json")}),
        );
    }
    println!(
        "{}",
        serde_json::to_string(&paths).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() -> Result<(), String> {
    run()
}

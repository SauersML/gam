//! Typed edit-response proposal coverage only; no model execution or fitting.
//! GRAMMAR.json FRESH_REPORT.json. Native-class matching is diagnostic only:
//! equations are never supplied to enumeration or used to admit candidates.
use gam_mpd::{
    composed_rule_search::Unary,
    operator_program::Interface,
    program_learned_dag::{self as learned, Expr, TypeRef},
};
use serde_json::json;
use std::{collections::BTreeSet, time::Instant};

fn subexpressions(e: &Expr, out: &mut BTreeSet<Expr>) {
    if !matches!(e, Expr::Argument(_)) {
        out.insert(e.clone());
    }
    match e {
        Expr::Argument(_) => {}
        Expr::Unary(_, x) | Expr::Affine { input: x, .. } => subexpressions(x, out),
        Expr::Binary(_, a, b) => {
            subexpressions(a, out);
            subexpressions(b, out);
        }
    }
}
fn nonlinear(e: &Expr) -> bool {
    match e {
        Expr::Argument(_) => false,
        Expr::Unary(_, _) => true,
        Expr::Affine { input, .. } => nonlinear(input),
        Expr::Binary(_, a, b) => nonlinear(a) || nonlinear(b),
    }
}
fn affine(e: &Expr) -> bool {
    match e {
        Expr::Argument(_) => false,
        Expr::Affine { .. } => true,
        Expr::Unary(_, x) => affine(x),
        Expr::Binary(_, a, b) => affine(a) || affine(b),
    }
}
fn shared_nonlinear(expressions: &[Expr]) -> bool {
    let mut common: Option<BTreeSet<Expr>> = None;
    for e in expressions {
        let mut own = BTreeSet::new();
        subexpressions(e, &mut own);
        common = Some(match common {
            None => own,
            Some(old) => old.intersection(&own).cloned().collect(),
        });
    }
    common.is_some_and(|set| {
        set.iter()
            .any(|e| matches!(e, Expr::Unary(_, _)) && affine(e))
    })
}
fn native_class(expressions: &[Expr], law: Unary) -> bool {
    if expressions.len() != 3 {
        return false;
    }
    let Expr::Affine {
        input: activation, ..
    } = &expressions[0]
    else {
        return false;
    };
    let Expr::Unary(actual_law, up) = activation.as_ref() else {
        return false;
    };
    if *actual_law != law {
        return false;
    }
    let Expr::Affine {
        output: TypeRef::Latent { width: 3072 },
        input,
        ..
    } = up.as_ref()
    else {
        return false;
    };
    if **input != Expr::Argument(0) {
        return false;
    }
    expressions[1..]
        .iter()
        .all(|e| matches!(e, Expr::Affine { input, .. } if input == activation))
}
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("GRAMMAR.json FRESH_REPORT.json".into());
    }
    let bytes = std::fs::read(&args[0]).map_err(|e| e.to_string())?;
    let settings: learned::Settings = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let inputs = vec![Interface::native(768).map_err(|e| e.to_string())?];
    let outputs = [768, 1, 1]
        .into_iter()
        .map(Interface::native)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let inventory = learned::enumerate_interfaces(&inputs, &outputs, &settings)?;
    let indices = |test: &dyn Fn(&learned::Proposal) -> bool| {
        inventory
            .proposals
            .iter()
            .enumerate()
            .filter_map(|(i, p)| test(p).then_some(i))
            .collect::<Vec<_>>()
    };
    let trainable_clean = indices(&|p| affine(&p.expressions[0]));
    let nonlinear_clean = indices(&|p| affine(&p.expressions[0]) && nonlinear(&p.expressions[0]));
    let all_shared = indices(&|p| shared_nonlinear(&p.expressions));
    let native = indices(&|p| native_class(&p.expressions, Unary::Gelu));
    let native_tanh = indices(&|p| native_class(&p.expressions, Unary::GeluTanh));
    let report = json!({
        "input_widths":[768], "output_widths":[768,1,1], "settings":settings,
        "seconds":start.elapsed().as_secs_f64(),
        "coverage":{"trainable_clean_indices":trainable_clean,
            "nonlinear_trainable_clean_indices":nonlinear_clean,
            "shared_learned_nonlinear_all_outputs_indices":all_shared,
            "native_gelu_3072_class_indices":native, "native_gelu_tanh_3072_class_indices":native_tanh},
        "inventory":inventory,
        "scope":"Bounded typed proposal enumeration only. Syntactic learned paths and shared nonlinear subexpressions are capacity diagnostics, not numerical sensitivity or fitted success. Native class matching only inspects generated formulas; it never supplies formulas or parameter sharing to enumeration. No model execution, matrix parameter initialization, fitting, GPU, or heldout data. Inventory order is its declared deterministic rank, not execution or fitness order."
    });
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut out, &report).map_err(|e| e.to_string())?;
    println!(
        "{} proposals; trainable clean {}; nonlinear clean {}; learned nonlinear shared all {}; native GELU3072 class {}; native GELU-tanh3072 class {}; {:.3}s",
        inventory.proposals.len(),
        trainable_clean.len(),
        nonlinear_clean.len(),
        all_shared.len(),
        native.len(),
        native_tanh.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

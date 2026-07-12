use proc_macro2::{Ident, Literal, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use std::collections::{HashMap, HashSet};
use syn::parse::{Parse, ParseStream};
use syn::{
    BinOp, Expr, ExprBinary, ExprCall, ExprGroup, ExprLit, ExprParen, ExprPath, ExprUnary, Lit,
    Path, Result, Token, UnOp, Visibility, braced, bracketed, parenthesized,
};

struct Leaf {
    alias: Ident,
    rust: Path,
    cuda: Ident,
}

enum RawStatement {
    Local {
        name: Ident,
        mutable: bool,
        value: Expr,
    },
    If {
        condition: Expr,
        assignments: Vec<(Ident, Expr)>,
    },
}

struct RawBody {
    statements: Vec<RawStatement>,
    result: Expr,
}

pub(crate) struct Input {
    visibility: Visibility,
    name: Ident,
    primaries: Vec<Ident>,
    constants: Vec<Ident>,
    leaves: Vec<Leaf>,
    witnesses: Vec<Ident>,
    body: RawBody,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let name = input.parse()?;

        let arguments;
        parenthesized!(arguments in input);
        let mut primaries = Vec::new();
        while !arguments.is_empty() && !arguments.peek(Token![;]) {
            primaries.push(arguments.parse::<Ident>()?);
            if arguments.peek(Token![,]) {
                arguments.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        let mut constants = Vec::new();
        if arguments.peek(Token![;]) {
            arguments.parse::<Token![;]>()?;
            while !arguments.is_empty() {
                constants.push(arguments.parse::<Ident>()?);
                if arguments.peek(Token![,]) {
                    arguments.parse::<Token![,]>()?;
                } else {
                    break;
                }
            }
        }
        if primaries.is_empty() {
            return Err(input.error("row_program requires at least one primary"));
        }
        if !arguments.is_empty() {
            return Err(arguments.error("invalid row_program argument list"));
        }

        let leaves_keyword = input.parse::<Ident>()?;
        if leaves_keyword != "leaves" {
            return Err(syn::Error::new_spanned(
                leaves_keyword,
                "row_program expects `leaves { ... }`",
            ));
        }
        let leaf_tokens;
        braced!(leaf_tokens in input);
        let mut leaves = Vec::new();
        while !leaf_tokens.is_empty() {
            let alias = leaf_tokens.parse()?;
            leaf_tokens.parse::<Token![=>]>()?;
            let rust = leaf_tokens.parse()?;
            leaf_tokens.parse::<Token![=>]>()?;
            let cuda = leaf_tokens.parse()?;
            leaves.push(Leaf { alias, rust, cuda });
            if leaf_tokens.peek(Token![,]) {
                leaf_tokens.parse::<Token![,]>()?;
            }
        }

        let witnesses_keyword = input.parse::<Ident>()?;
        if witnesses_keyword != "witnesses" {
            return Err(syn::Error::new_spanned(
                witnesses_keyword,
                "row_program expects `witnesses [ ... ]`",
            ));
        }
        let witness_tokens;
        bracketed!(witness_tokens in input);
        let mut witnesses = Vec::new();
        while !witness_tokens.is_empty() {
            witnesses.push(witness_tokens.parse()?);
            if witness_tokens.peek(Token![,]) {
                witness_tokens.parse::<Token![,]>()?;
            }
        }
        input.parse::<Token![;]>()?;

        let body_tokens;
        braced!(body_tokens in input);
        let mut statements = Vec::new();
        let mut result = None;
        while !body_tokens.is_empty() {
            if body_tokens.peek(Token![let]) {
                body_tokens.parse::<Token![let]>()?;
                let mutable = if body_tokens.peek(Token![mut]) {
                    body_tokens.parse::<Token![mut]>()?;
                    true
                } else {
                    false
                };
                let name = body_tokens.parse()?;
                body_tokens.parse::<Token![=]>()?;
                let value = body_tokens.parse()?;
                body_tokens.parse::<Token![;]>()?;
                statements.push(RawStatement::Local {
                    name,
                    mutable,
                    value,
                });
                continue;
            }
            if body_tokens.peek(Token![if]) {
                body_tokens.parse::<Token![if]>()?;
                let condition_tokens;
                parenthesized!(condition_tokens in body_tokens);
                let condition = condition_tokens.parse()?;
                if !condition_tokens.is_empty() {
                    return Err(condition_tokens.error("invalid row_program condition"));
                }
                let assignment_tokens;
                braced!(assignment_tokens in body_tokens);
                let mut assignments = Vec::new();
                while !assignment_tokens.is_empty() {
                    let target = assignment_tokens.parse()?;
                    assignment_tokens.parse::<Token![=]>()?;
                    let value = assignment_tokens.parse()?;
                    assignment_tokens.parse::<Token![;]>()?;
                    assignments.push((target, value));
                }
                statements.push(RawStatement::If {
                    condition,
                    assignments,
                });
                continue;
            }
            if body_tokens.peek(Token![return]) {
                body_tokens.parse::<Token![return]>()?;
                if result.is_some() {
                    return Err(body_tokens.error("row_program has more than one return"));
                }
                result = Some(body_tokens.parse()?);
                body_tokens.parse::<Token![;]>()?;
                if !body_tokens.is_empty() {
                    return Err(body_tokens.error("row_program return must be last"));
                }
                continue;
            }
            return Err(body_tokens.error("row_program supports only let, if, and return"));
        }
        let result = result.ok_or_else(|| input.error("row_program requires a final return"))?;

        Ok(Self {
            visibility,
            name,
            primaries,
            constants,
            leaves,
            witnesses,
            body: RawBody { statements, result },
        })
    }
}

#[derive(Clone)]
enum ProgramExpr {
    Path(Ident),
    Zero,
    Neg(Box<ProgramExpr>),
    Scale(Box<ProgramExpr>, Expr),
    AddConstant(Box<ProgramExpr>, Expr),
    Add(Box<ProgramExpr>, Box<ProgramExpr>),
    Mul(Box<ProgramExpr>, Box<ProgramExpr>),
    Compose {
        leaf: usize,
        value: Ident,
        arguments: Vec<Expr>,
    },
}

enum Statement {
    Local {
        name: Ident,
        mutable: bool,
        value: ProgramExpr,
    },
    If {
        condition: Expr,
        assignments: Vec<(Ident, ProgramExpr)>,
    },
}

fn bare_call_name(call: &ExprCall) -> Result<&Ident> {
    let Expr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new_spanned(
            &call.func,
            "row_program operations must use bare function names",
        ));
    };
    path.path.get_ident().ok_or_else(|| {
        syn::Error::new_spanned(
            &call.func,
            "row_program operations must use bare function names",
        )
    })
}

fn path_ident(path: &ExprPath) -> Result<&Ident> {
    path.path
        .get_ident()
        .ok_or_else(|| syn::Error::new_spanned(path, "row_program paths must be identifiers"))
}

fn numeric_literal(literal: &ExprLit) -> bool {
    matches!(&literal.lit, Lit::Float(_) | Lit::Int(_))
}

fn validate_scalar(expression: &Expr, constants: &HashSet<String>) -> Result<()> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if constants.contains(&ident.to_string()) {
                Ok(())
            } else {
                Err(syn::Error::new_spanned(
                    ident,
                    format!("unknown row_program scalar `{ident}`"),
                ))
            }
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(()),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            validate_scalar(expr, constants)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => validate_scalar(expr, constants),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) if matches!(
            op,
            BinOp::Add(_)
                | BinOp::Sub(_)
                | BinOp::Mul(_)
                | BinOp::Div(_)
                | BinOp::Eq(_)
                | BinOp::Ne(_)
                | BinOp::Lt(_)
                | BinOp::Le(_)
                | BinOp::Gt(_)
                | BinOp::Ge(_)
        ) =>
        {
            validate_scalar(left, constants)?;
            validate_scalar(right, constants)
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_program scalar expression",
        )),
    }
}

fn parse_program_expr(
    expression: &Expr,
    bindings: &HashSet<String>,
    constants: &HashSet<String>,
    leaves: &HashMap<String, usize>,
) -> Result<ProgramExpr> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if bindings.contains(&ident.to_string()) {
                Ok(ProgramExpr::Path(ident.clone()))
            } else {
                Err(syn::Error::new_spanned(
                    ident,
                    format!("unknown row_program jet `{ident}`"),
                ))
            }
        }
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            parse_program_expr(expr, bindings, constants, leaves)
        }
        Expr::Call(call) => {
            let operation = bare_call_name(call)?.to_string();
            let arguments = call.args.iter().collect::<Vec<_>>();
            match operation.as_str() {
                "zero" if arguments.is_empty() => Ok(ProgramExpr::Zero),
                "neg" if arguments.len() == 1 => Ok(ProgramExpr::Neg(Box::new(
                    parse_program_expr(arguments[0], bindings, constants, leaves)?,
                ))),
                "scale" | "add_constant" if arguments.len() == 2 => {
                    let value = parse_program_expr(arguments[0], bindings, constants, leaves)?;
                    validate_scalar(arguments[1], constants)?;
                    if operation == "scale" {
                        Ok(ProgramExpr::Scale(Box::new(value), arguments[1].clone()))
                    } else {
                        Ok(ProgramExpr::AddConstant(
                            Box::new(value),
                            arguments[1].clone(),
                        ))
                    }
                }
                "add" | "mul" if arguments.len() == 2 => {
                    let left = parse_program_expr(arguments[0], bindings, constants, leaves)?;
                    let right = parse_program_expr(arguments[1], bindings, constants, leaves)?;
                    if operation == "add" {
                        Ok(ProgramExpr::Add(Box::new(left), Box::new(right)))
                    } else {
                        Ok(ProgramExpr::Mul(Box::new(left), Box::new(right)))
                    }
                }
                "compose" if arguments.len() >= 2 => {
                    let Expr::Path(leaf_path) = arguments[0] else {
                        return Err(syn::Error::new_spanned(
                            arguments[0],
                            "row_program compose leaf must be an identifier",
                        ));
                    };
                    let leaf_ident = path_ident(leaf_path)?;
                    let leaf = leaves
                        .get(&leaf_ident.to_string())
                        .copied()
                        .ok_or_else(|| {
                            syn::Error::new_spanned(
                                leaf_ident,
                                format!("unknown row_program leaf `{leaf_ident}`"),
                            )
                        })?;
                    let Expr::Path(value_path) = arguments[1] else {
                        return Err(syn::Error::new_spanned(
                            arguments[1],
                            "row_program compose value must be a named jet",
                        ));
                    };
                    let value = path_ident(value_path)?.clone();
                    if !bindings.contains(&value.to_string()) {
                        return Err(syn::Error::new_spanned(
                            value,
                            "row_program compose value is not defined",
                        ));
                    }
                    let mut scalar_arguments = Vec::new();
                    for argument in &arguments[2..] {
                        validate_scalar(argument, constants)?;
                        scalar_arguments.push((*argument).clone());
                    }
                    Ok(ProgramExpr::Compose {
                        leaf,
                        value,
                        arguments: scalar_arguments,
                    })
                }
                _ => Err(syn::Error::new_spanned(
                    call,
                    format!(
                        "unsupported row_program operation `{operation}` or wrong argument count"
                    ),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "row_program jet expressions use only named jets and explicit operations",
        )),
    }
}

fn rust_expression(expression: &ProgramExpr, leaves: &[Leaf]) -> TokenStream2 {
    match expression {
        ProgramExpr::Path(ident) => quote!(#ident),
        ProgramExpr::Zero => quote!(S::constant(0.0)),
        ProgramExpr::Neg(value) => {
            let value = rust_expression(value, leaves);
            quote!({ let value = #value; value.neg() })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = rust_expression(value, leaves);
            quote!({ let value = #value; value.scale(#scalar) })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let value = rust_expression(value, leaves);
            quote!({ let value = #value; value.add(&S::constant(#scalar)) })
        }
        ProgramExpr::Add(left, right) => {
            let left = rust_expression(left, leaves);
            let right = rust_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.add(&right) })
        }
        ProgramExpr::Mul(left, right) => {
            let left = rust_expression(left, leaves);
            let right = rust_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.mul(&right) })
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let value_ident = value;
            let rust_leaf = &leaves[*leaf].rust;
            quote!({
                let value = #value_ident;
                value.compose_unary(#rust_leaf(value.value(), #(#arguments),*))
            })
        }
    }
}

fn rust_scalar_expression(expression: &ProgramExpr, leaves: &[Leaf]) -> TokenStream2 {
    match expression {
        ProgramExpr::Path(ident) => quote!(#ident),
        ProgramExpr::Zero => quote!(0.0),
        ProgramExpr::Neg(value) => {
            let value = rust_scalar_expression(value, leaves);
            quote!(-(#value))
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = rust_scalar_expression(value, leaves);
            quote!((#value) * (#scalar))
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let value = rust_scalar_expression(value, leaves);
            quote!((#value) + (#scalar))
        }
        ProgramExpr::Add(left, right) => {
            let left = rust_scalar_expression(left, leaves);
            let right = rust_scalar_expression(right, leaves);
            quote!((#left) + (#right))
        }
        ProgramExpr::Mul(left, right) => {
            let left = rust_scalar_expression(left, leaves);
            let right = rust_scalar_expression(right, leaves);
            quote!((#left) * (#right))
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let rust_leaf = &leaves[*leaf].rust;
            quote!(#rust_leaf(#value, #(#arguments),*)[0])
        }
    }
}

fn collect_dependencies(expression: &ProgramExpr, dependencies: &mut HashSet<String>) {
    match expression {
        ProgramExpr::Path(ident) => {
            dependencies.insert(ident.to_string());
        }
        ProgramExpr::Zero => {}
        ProgramExpr::Neg(value)
        | ProgramExpr::Scale(value, _)
        | ProgramExpr::AddConstant(value, _) => collect_dependencies(value, dependencies),
        ProgramExpr::Add(left, right) | ProgramExpr::Mul(left, right) => {
            collect_dependencies(left, dependencies);
            collect_dependencies(right, dependencies);
        }
        ProgramExpr::Compose { value, .. } => {
            dependencies.insert(value.to_string());
        }
    }
}

fn witness_dependencies(statements: &[Statement], witnesses: &[Ident]) -> HashSet<String> {
    let mut dependencies = witnesses
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    loop {
        let previous_len = dependencies.len();
        for statement in statements.iter().rev() {
            match statement {
                Statement::Local { name, value, .. } => {
                    if dependencies.contains(&name.to_string()) {
                        collect_dependencies(value, &mut dependencies);
                    }
                }
                Statement::If { assignments, .. } => {
                    for (target, value) in assignments {
                        if dependencies.contains(&target.to_string()) {
                            collect_dependencies(value, &mut dependencies);
                        }
                    }
                }
            }
        }
        if dependencies.len() == previous_len {
            return dependencies;
        }
    }
}

fn collect_scalar_expression_dependencies(expression: &Expr, dependencies: &mut HashSet<String>) {
    match expression {
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                dependencies.insert(ident.to_string());
            }
        }
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            collect_scalar_expression_dependencies(expr, dependencies);
        }
        Expr::Unary(ExprUnary { expr, .. }) => {
            collect_scalar_expression_dependencies(expr, dependencies);
        }
        Expr::Binary(ExprBinary { left, right, .. }) => {
            collect_scalar_expression_dependencies(left, dependencies);
            collect_scalar_expression_dependencies(right, dependencies);
        }
        Expr::Lit(_) => {}
        _ => unreachable!("validated row_program scalar grammar"),
    }
}

fn collect_program_scalar_dependencies(
    expression: &ProgramExpr,
    dependencies: &mut HashSet<String>,
) {
    match expression {
        ProgramExpr::Path(_) | ProgramExpr::Zero => {}
        ProgramExpr::Neg(value) => collect_program_scalar_dependencies(value, dependencies),
        ProgramExpr::Scale(value, scalar) | ProgramExpr::AddConstant(value, scalar) => {
            collect_program_scalar_dependencies(value, dependencies);
            collect_scalar_expression_dependencies(scalar, dependencies);
        }
        ProgramExpr::Add(left, right) | ProgramExpr::Mul(left, right) => {
            collect_program_scalar_dependencies(left, dependencies);
            collect_program_scalar_dependencies(right, dependencies);
        }
        ProgramExpr::Compose { arguments, .. } => {
            for argument in arguments {
                collect_scalar_expression_dependencies(argument, dependencies);
            }
        }
    }
}

fn witness_scalar_dependencies(
    statements: &[Statement],
    jet_dependencies: &HashSet<String>,
) -> HashSet<String> {
    let mut dependencies = HashSet::new();
    for statement in statements {
        match statement {
            Statement::Local { name, value, .. }
                if jet_dependencies.contains(&name.to_string()) =>
            {
                collect_program_scalar_dependencies(value, &mut dependencies);
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut condition_is_needed = false;
                for (target, value) in assignments {
                    if jet_dependencies.contains(&target.to_string()) {
                        collect_program_scalar_dependencies(value, &mut dependencies);
                        condition_is_needed = true;
                    }
                }
                if condition_is_needed {
                    collect_scalar_expression_dependencies(condition, &mut dependencies);
                }
            }
            Statement::Local { .. } => {}
        }
    }
    dependencies
}

fn scalar_cuda(expression: &Expr, constants: &HashSet<String>) -> Result<String> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if constants.contains(&ident.to_string()) {
                Ok(format!("in.{ident}"))
            } else {
                Err(syn::Error::new_spanned(ident, "unknown CUDA scalar"))
            }
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(quote!(#literal).to_string()),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            Ok(format!("({})", scalar_cuda(expr, constants)?))
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => Ok(format!("-({})", scalar_cuda(expr, constants)?)),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let operator = match op {
                BinOp::Add(_) => "+",
                BinOp::Sub(_) => "-",
                BinOp::Mul(_) => "*",
                BinOp::Div(_) => "/",
                BinOp::Eq(_) => "==",
                BinOp::Ne(_) => "!=",
                BinOp::Lt(_) => "<",
                BinOp::Le(_) => "<=",
                BinOp::Gt(_) => ">",
                BinOp::Ge(_) => ">=",
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "unsupported CUDA scalar operator",
                    ));
                }
            };
            Ok(format!(
                "({} {operator} {})",
                scalar_cuda(left, constants)?,
                scalar_cuda(right, constants)?
            ))
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported CUDA scalar expression",
        )),
    }
}

#[derive(Clone)]
struct CudaJet {
    value: String,
    gradient: Vec<Option<String>>,
    // Only entries with a <= b are populated. The generated CUDA computes the
    // packed triangle once and scatters it symmetrically at the output seam.
    hessian: Vec<Option<String>>,
}

#[derive(Clone)]
struct CudaSupport {
    gradient: Vec<bool>,
    hessian: Vec<bool>,
}

impl CudaSupport {
    fn empty(dimension: usize) -> Self {
        Self {
            gradient: vec![false; dimension],
            hessian: vec![false; dimension * dimension],
        }
    }

    fn include(&mut self, jet: &CudaJet) {
        for (present, component) in self.gradient.iter_mut().zip(&jet.gradient) {
            *present |= component.is_some();
        }
        for (present, component) in self.hessian.iter_mut().zip(&jet.hessian) {
            *present |= component.is_some();
        }
    }
}

impl CudaJet {
    fn zero(dimension: usize) -> Self {
        Self {
            value: "0.0".to_string(),
            gradient: vec![None; dimension],
            hessian: vec![None; dimension * dimension],
        }
    }

    fn primary(name: &str, axis: usize, dimension: usize) -> Self {
        let mut out = Self::zero(dimension);
        out.value = name.to_string();
        out.gradient[axis] = Some("1.0".to_string());
        out
    }

    fn support(&self) -> CudaSupport {
        let mut support = CudaSupport::empty(self.gradient.len());
        support.include(self);
        support
    }

    fn reference(name: &str, support: &CudaSupport, dimension: usize) -> Self {
        let mut out = Self::zero(dimension);
        out.value = format!("{name}_v");
        for axis in 0..dimension {
            if support.gradient[axis] {
                out.gradient[axis] = Some(format!("{name}_g{axis}"));
            }
            for other in axis..dimension {
                let index = axis * dimension + other;
                if support.hessian[index] {
                    out.hessian[index] = Some(format!("{name}_h{axis}_{other}"));
                }
            }
        }
        out
    }
}

fn cuda_is_zero(value: &str) -> bool {
    value == "0.0"
}

fn cuda_is_one(value: &str) -> bool {
    value == "1.0"
}

fn cuda_is_negative_one(value: &str) -> bool {
    matches!(value, "-1.0" | "-(1.0)" | "(-1.0)")
}

fn cuda_negate(value: &str) -> String {
    if cuda_is_zero(value) {
        "0.0".to_string()
    } else if cuda_is_negative_one(value) {
        "1.0".to_string()
    } else if cuda_is_one(value) {
        "-1.0".to_string()
    } else {
        format!("-({value})")
    }
}

fn cuda_add(left: &str, right: &str) -> String {
    if cuda_is_zero(left) {
        right.to_string()
    } else if cuda_is_zero(right) {
        left.to_string()
    } else {
        format!("({left} + {right})")
    }
}

fn cuda_multiply(left: &str, right: &str) -> String {
    if cuda_is_zero(left) || cuda_is_zero(right) {
        "0.0".to_string()
    } else if cuda_is_one(left) {
        right.to_string()
    } else if cuda_is_one(right) {
        left.to_string()
    } else if cuda_is_negative_one(left) {
        cuda_negate(right)
    } else if cuda_is_negative_one(right) {
        cuda_negate(left)
    } else {
        format!("({left} * {right})")
    }
}

fn cuda_add_component(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(cuda_add(left, right)),
        (Some(value), None) | (None, Some(value)) => Some(value.clone()),
        (None, None) => None,
    }
}

fn cuda_multiply_component(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(cuda_multiply(left, right)),
        _ => None,
    }
}

fn cuda_scale_component(component: &Option<String>, scalar: &str) -> Option<String> {
    component
        .as_ref()
        .map(|component| cuda_multiply(component, scalar))
}

fn cuda_add_jets(left: CudaJet, right: CudaJet) -> CudaJet {
    CudaJet {
        value: cuda_add(&left.value, &right.value),
        gradient: left
            .gradient
            .iter()
            .zip(&right.gradient)
            .map(|(left, right)| cuda_add_component(left, right))
            .collect(),
        hessian: left
            .hessian
            .iter()
            .zip(&right.hessian)
            .map(|(left, right)| cuda_add_component(left, right))
            .collect(),
    }
}

fn cuda_multiply_jets(left: CudaJet, right: CudaJet) -> CudaJet {
    let dimension = left.gradient.len();
    let mut gradient = vec![None; dimension];
    let mut hessian = vec![None; dimension * dimension];
    for axis in 0..dimension {
        gradient[axis] = cuda_add_component(
            &cuda_scale_component(&right.gradient[axis], &left.value),
            &cuda_scale_component(&left.gradient[axis], &right.value),
        );
        for other in axis..dimension {
            let index = axis * dimension + other;
            let inherited_right = cuda_scale_component(&right.hessian[index], &left.value);
            let cross_forward =
                cuda_multiply_component(&left.gradient[axis], &right.gradient[other]);
            let cross_reverse =
                cuda_multiply_component(&left.gradient[other], &right.gradient[axis]);
            let inherited_left = cuda_scale_component(&left.hessian[index], &right.value);
            hessian[index] = cuda_add_component(
                &cuda_add_component(
                    &cuda_add_component(&inherited_right, &cross_forward),
                    &cross_reverse,
                ),
                &inherited_left,
            );
        }
    }
    CudaJet {
        value: cuda_multiply(&left.value, &right.value),
        gradient,
        hessian,
    }
}

fn cuda_expression(
    expression: &ProgramExpr,
    owner: &str,
    leaves: &[Leaf],
    constants: &HashSet<String>,
    bindings: &HashMap<String, CudaJet>,
    dimension: usize,
    stack_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<CudaJet> {
    let mut child = |expression: &ProgramExpr| {
        cuda_expression(
            expression,
            owner,
            leaves,
            constants,
            bindings,
            dimension,
            stack_index,
            preludes,
        )
    };
    match expression {
        ProgramExpr::Path(ident) => bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
            syn::Error::new_spanned(ident, "CUDA row_program binding is not defined")
        }),
        ProgramExpr::Zero => Ok(CudaJet::zero(dimension)),
        ProgramExpr::Neg(value) => {
            let value = child(value)?;
            Ok(CudaJet {
                value: cuda_negate(&value.value),
                gradient: value
                    .gradient
                    .iter()
                    .map(|component| component.as_ref().map(|value| cuda_negate(value)))
                    .collect(),
                hessian: value
                    .hessian
                    .iter()
                    .map(|component| component.as_ref().map(|value| cuda_negate(value)))
                    .collect(),
            })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = child(value)?;
            let scalar = scalar_cuda(scalar, constants)?;
            Ok(CudaJet {
                value: cuda_multiply(&value.value, &scalar),
                gradient: value
                    .gradient
                    .iter()
                    .map(|component| cuda_scale_component(component, &scalar))
                    .collect(),
                hessian: value
                    .hessian
                    .iter()
                    .map(|component| cuda_scale_component(component, &scalar))
                    .collect(),
            })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            value.value = cuda_add(&value.value, &scalar_cuda(scalar, constants)?);
            Ok(value)
        }
        ProgramExpr::Add(left, right) => Ok(cuda_add_jets(child(left)?, child(right)?)),
        ProgramExpr::Mul(left, right) => Ok(cuda_multiply_jets(child(left)?, child(right)?)),
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "CUDA compose input is not defined")
            })?;
            let suffix = *stack_index;
            *stack_index += 1;
            let stack = format!("{owner}_stack{suffix}");
            let cuda_leaf = &leaves[*leaf].cuda;
            let mut leaf_arguments = vec![input.value.clone()];
            for argument in arguments {
                leaf_arguments.push(scalar_cuda(argument, constants)?);
            }
            leaf_arguments.push(stack.clone());
            preludes.push(format!(
                "double {stack}[3];\n{cuda_leaf}({});",
                leaf_arguments.join(", ")
            ));

            let first = format!("{stack}[1]");
            let second = format!("{stack}[2]");
            let mut gradient = vec![None; dimension];
            let mut hessian = vec![None; dimension * dimension];
            for axis in 0..dimension {
                gradient[axis] = cuda_scale_component(&input.gradient[axis], &first);
                for other in axis..dimension {
                    let index = axis * dimension + other;
                    let inherited = cuda_scale_component(&input.hessian[index], &first);
                    let curvature =
                        cuda_multiply_component(&input.gradient[axis], &input.gradient[other])
                            .map(|component| cuda_multiply(&second, &component));
                    hessian[index] = cuda_add_component(&inherited, &curvature);
                }
            }
            Ok(CudaJet {
                value: format!("{stack}[0]"),
                gradient,
                hessian,
            })
        }
    }
}

struct CudaLocal {
    name: String,
    mutable: bool,
    value: CudaJet,
    preludes: Vec<String>,
}

struct CudaAssignment {
    target: String,
    value: CudaJet,
    preludes: Vec<String>,
}

enum CudaStatement {
    Local(CudaLocal),
    If {
        condition: String,
        assignments: Vec<CudaAssignment>,
    },
}

fn cuda_push_preludes(source: &mut String, preludes: &[String], indentation: &str) {
    for prelude in preludes {
        for line in prelude.lines() {
            source.push_str(indentation);
            source.push_str(line);
            source.push('\n');
        }
    }
}

fn cuda_component(component: &Option<String>) -> &str {
    component.as_deref().unwrap_or("0.0")
}

fn cuda_source(
    name: &Ident,
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
) -> Result<String> {
    let dimension = primaries.len();
    let parameters = primaries
        .iter()
        .map(|primary| format!("double {primary}"))
        .chain([
            "const RowIn& in".to_string(),
            "double* row_value".to_string(),
            "double* row_gradient".to_string(),
            "double* row_hessian".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let mut bindings = HashMap::<String, CudaJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            CudaJet::primary(&primary.to_string(), axis, dimension),
        );
    }
    let mut mutable_support = HashMap::<String, CudaSupport>::new();
    let mut cuda_statements = Vec::new();
    // One source-wide namespace makes temporary declarations collision-free,
    // including repeated assignments to the same mutable local in one scope.
    let mut stack_index = 0;
    for statement in statements {
        match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let mut preludes = Vec::new();
                let value = cuda_expression(
                    value,
                    &name.to_string(),
                    leaves,
                    constants,
                    &bindings,
                    dimension,
                    &mut stack_index,
                    &mut preludes,
                )?;
                let support = value.support();
                if *mutable {
                    mutable_support.insert(name.to_string(), support.clone());
                }
                bindings.insert(
                    name.to_string(),
                    CudaJet::reference(&name.to_string(), &support, dimension),
                );
                cuda_statements.push(CudaStatement::Local(CudaLocal {
                    name: name.to_string(),
                    mutable: *mutable,
                    value,
                    preludes,
                }));
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut cuda_assignments = Vec::new();
                for (target, value) in assignments {
                    let mut preludes = Vec::new();
                    let value = cuda_expression(
                        value,
                        &target.to_string(),
                        leaves,
                        constants,
                        &bindings,
                        dimension,
                        &mut stack_index,
                        &mut preludes,
                    )?;
                    let support = mutable_support
                        .get_mut(&target.to_string())
                        .expect("validated mutable CUDA target");
                    support.include(&value);
                    bindings.insert(
                        target.to_string(),
                        CudaJet::reference(&target.to_string(), support, dimension),
                    );
                    cuda_assignments.push(CudaAssignment {
                        target: target.to_string(),
                        value,
                        preludes,
                    });
                }
                cuda_statements.push(CudaStatement::If {
                    condition: scalar_cuda(condition, constants)?,
                    assignments: cuda_assignments,
                });
            }
        }
    }
    let mut preludes = Vec::new();
    let result = cuda_expression(
        result,
        "result",
        leaves,
        constants,
        &bindings,
        dimension,
        &mut stack_index,
        &mut preludes,
    )?;

    let mut source = format!("__device__ __forceinline__ void {name}(\n        {parameters}) {{\n");
    for statement in &cuda_statements {
        match statement {
            CudaStatement::Local(local) => {
                cuda_push_preludes(&mut source, &local.preludes, "    ");
                let support = if local.mutable {
                    mutable_support
                        .get(&local.name)
                        .expect("mutable CUDA support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                source.push_str(&format!(
                    "    double {}_v = {};\n",
                    local.name, local.value.value
                ));
                for axis in 0..dimension {
                    if support.gradient[axis] {
                        source.push_str(&format!(
                            "    double {}_g{axis} = {};\n",
                            local.name,
                            cuda_component(&local.value.gradient[axis]),
                        ));
                    }
                    for other in axis..dimension {
                        let index = axis * dimension + other;
                        if support.hessian[index] {
                            source.push_str(&format!(
                                "    double {}_h{axis}_{other} = {};\n",
                                local.name,
                                cuda_component(&local.value.hessian[index]),
                            ));
                        }
                    }
                }
            }
            CudaStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if ({condition}) {{\n"));
                for assignment in assignments {
                    cuda_push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = mutable_support
                        .get(&assignment.target)
                        .expect("mutable CUDA assignment support exists");
                    source.push_str(&format!(
                        "        {}_v = {};\n",
                        assignment.target, assignment.value.value,
                    ));
                    for axis in 0..dimension {
                        if support.gradient[axis] {
                            source.push_str(&format!(
                                "        {}_g{axis} = {};\n",
                                assignment.target,
                                cuda_component(&assignment.value.gradient[axis]),
                            ));
                        }
                        for other in axis..dimension {
                            let index = axis * dimension + other;
                            if support.hessian[index] {
                                source.push_str(&format!(
                                    "        {}_h{axis}_{other} = {};\n",
                                    assignment.target,
                                    cuda_component(&assignment.value.hessian[index]),
                                ));
                            }
                        }
                    }
                }
                source.push_str("    }\n");
            }
        }
    }
    cuda_push_preludes(&mut source, &preludes, "    ");
    source.push_str(&format!("    *row_value = {};\n", result.value));
    for axis in 0..dimension {
        source.push_str(&format!(
            "    row_gradient[{axis}] = {};\n",
            cuda_component(&result.gradient[axis]),
        ));
        for other in axis..dimension {
            let index = axis * dimension + other;
            let component = cuda_component(&result.hessian[index]);
            source.push_str(&format!(
                "    row_hessian[{}] = {component};\n",
                axis * dimension + other,
            ));
            if axis != other {
                source.push_str(&format!(
                    "    row_hessian[{}] = {component};\n",
                    other * dimension + axis,
                ));
            }
        }
    }
    source.push_str("}\n");
    Ok(source)
}

pub(crate) fn expand(input: Input) -> Result<TokenStream2> {
    let Input {
        visibility,
        name,
        primaries,
        constants,
        leaves,
        witnesses,
        body,
    } = input;

    let mut all_names = HashSet::new();
    for name in primaries.iter().chain(constants.iter()) {
        if !all_names.insert(name.to_string()) {
            return Err(syn::Error::new_spanned(
                name,
                "row_program argument names must be unique",
            ));
        }
    }
    let constant_names = constants
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    let mut leaf_indices = HashMap::new();
    for (index, leaf) in leaves.iter().enumerate() {
        if leaf_indices.insert(leaf.alias.to_string(), index).is_some() {
            return Err(syn::Error::new_spanned(
                &leaf.alias,
                "row_program leaf aliases must be unique",
            ));
        }
    }

    let mut bindings = primaries
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    let mut mutable = HashSet::new();
    let mut statements = Vec::new();
    for raw in body.statements {
        match raw {
            RawStatement::Local {
                name,
                mutable: is_mutable,
                value,
            } => {
                if all_names.contains(&name.to_string()) || bindings.contains(&name.to_string()) {
                    return Err(syn::Error::new_spanned(
                        name,
                        "row_program local name is already defined",
                    ));
                }
                let value = parse_program_expr(&value, &bindings, &constant_names, &leaf_indices)?;
                bindings.insert(name.to_string());
                if is_mutable {
                    mutable.insert(name.to_string());
                }
                statements.push(Statement::Local {
                    name,
                    mutable: is_mutable,
                    value,
                });
            }
            RawStatement::If {
                condition,
                assignments,
            } => {
                validate_scalar(&condition, &constant_names)?;
                let mut parsed_assignments = Vec::new();
                for (target, value) in assignments {
                    if !mutable.contains(&target.to_string()) {
                        return Err(syn::Error::new_spanned(
                            target,
                            "row_program assignment target must be a mutable local",
                        ));
                    }
                    parsed_assignments.push((
                        target,
                        parse_program_expr(&value, &bindings, &constant_names, &leaf_indices)?,
                    ));
                }
                statements.push(Statement::If {
                    condition,
                    assignments: parsed_assignments,
                });
            }
        }
    }
    let result = parse_program_expr(&body.result, &bindings, &constant_names, &leaf_indices)?;
    for witness in &witnesses {
        if !bindings.contains(&witness.to_string()) {
            return Err(syn::Error::new_spanned(
                witness,
                "row_program witness is not a defined jet",
            ));
        }
    }

    let rust_statements = statements.iter().map(|statement| match statement {
        Statement::Local {
            name,
            mutable,
            value,
        } => {
            let value = rust_expression(value, &leaves);
            if *mutable {
                quote!(let mut #name = #value;)
            } else {
                quote!(let #name = #value;)
            }
        }
        Statement::If {
            condition,
            assignments,
        } => {
            let assignments = assignments.iter().map(|(target, value)| {
                let value = rust_expression(value, &leaves);
                quote!(#target = #value;)
            });
            quote!(if #condition { #(#assignments)* })
        }
    });
    let rust_result = rust_expression(&result, &leaves);
    let witness_values = witnesses.iter().map(|witness| quote!(#witness.value()));
    let witness_count = witnesses.len();
    let scalar_witness_dependencies = witness_dependencies(&statements, &witnesses);
    let scalar_witness_scalar_dependencies =
        witness_scalar_dependencies(&statements, &scalar_witness_dependencies);
    let scalar_witness_statements = statements.iter().filter_map(|statement| match statement {
        Statement::Local {
            name,
            mutable,
            value,
        } if scalar_witness_dependencies.contains(&name.to_string()) => {
            let value = rust_scalar_expression(value, &leaves);
            Some(if *mutable {
                quote!(let mut #name = #value;)
            } else {
                quote!(let #name = #value;)
            })
        }
        Statement::If {
            condition,
            assignments,
        } => {
            let assignments = assignments
                .iter()
                .filter(|(target, _)| scalar_witness_dependencies.contains(&target.to_string()))
                .map(|(target, value)| {
                    let value = rust_scalar_expression(value, &leaves);
                    quote!(#target = #value;)
                })
                .collect::<Vec<_>>();
            (!assignments.is_empty()).then(|| quote!(if #condition { #(#assignments)* }))
        }
        Statement::Local { .. } => None,
    });
    let scalar_witness_name = format_ident!("{}_witnesses", name);
    let scalar_witness_function = if witnesses.is_empty() {
        quote!()
    } else {
        let scalar_witness_primaries = primaries
            .iter()
            .filter(|primary| scalar_witness_dependencies.contains(&primary.to_string()));
        let scalar_witness_constants = constants
            .iter()
            .filter(|constant| scalar_witness_scalar_dependencies.contains(&constant.to_string()));
        let scalar_witness_values = witnesses.iter();
        quote! {
            #[inline(always)]
            #visibility fn #scalar_witness_name(
                #(#scalar_witness_primaries: f64,)*
                #(#scalar_witness_constants: f64),*
            ) -> [f64; #witness_count] {
                #(#scalar_witness_statements)*
                [#(#scalar_witness_values),*]
            }
        }
    };
    let dimension = primaries.len();
    let cuda = cuda_source(
        &name,
        &primaries,
        &constant_names,
        &leaves,
        &statements,
        &result,
    )?;
    let cuda_literal = Literal::string(&cuda);
    let cuda_name = format_ident!("{}_CUDA_VGH", name.to_string().to_uppercase());

    Ok(quote! {
        #[inline(always)]
        #visibility fn #name<S: ::gam_math::jet_scalar::JetScalar<#dimension>>(
            #(#primaries: &S,)*
            #(#constants: f64),*
        ) -> (S, [f64; #witness_count]) {
            #(#rust_statements)*
            let __row_program_result = #rust_result;
            (__row_program_result, [#(#witness_values),*])
        }

        #scalar_witness_function

        #visibility const #cuda_name: &str = #cuda_literal;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn emitted_cuda(input: TokenStream2) -> String {
        let input = syn::parse2::<Input>(input).expect("parse row program");
        let expanded = expand(input).expect("expand row program");
        let file = syn::parse2::<syn::File>(expanded).expect("parse macro expansion");
        file.items
            .into_iter()
            .find_map(|item| {
                let syn::Item::Const(item) = item else {
                    return None;
                };
                let syn::Expr::Lit(expression) = *item.expr else {
                    return None;
                };
                let syn::Lit::Str(source) = expression.lit else {
                    return None;
                };
                Some(source.value())
            })
            .expect("expanded CUDA source constant")
    }

    fn emitted_function(input: TokenStream2, name: &str) -> String {
        let input = syn::parse2::<Input>(input).expect("parse row program");
        let expanded = expand(input).expect("expand row program");
        let file = syn::parse2::<syn::File>(expanded).expect("parse macro expansion");
        file.items
            .into_iter()
            .find_map(|item| {
                let syn::Item::Fn(item) = item else {
                    return None;
                };
                (item.sig.ident == name).then(|| quote!(#item).to_string())
            })
            .expect("expanded function")
    }

    #[test]
    fn emits_one_generic_and_cuda_schedule_with_branch_and_leaves() {
        let input = syn::parse2::<Input>(quote! {
            pub(crate) fn sample(q, g; weight, event, scale)
            leaves {
                sqrt => sqrt_stack => d_sqrt,
                log => log_stack => d_log,
            }
            witnesses [adjusted];
            {
                let scaled = scale(g, scale);
                let square = add_constant(mul(scaled, scaled), 1.0);
                let correction = compose(sqrt, square);
                let adjusted = mul(q, correction);
                let mut event_term = zero();
                if (event > 0.0) {
                    event_term = scale(compose(log, adjusted), -(weight * event));
                }
                return add(adjusted, event_term);
            }
        })
        .expect("parse row program");
        let expanded = expand(input).expect("expand row program").to_string();
        assert!(expanded.contains("JetScalar < 2usize >"));
        assert!(expanded.contains("SAMPLE_CUDA_VGH"));
        assert!(expanded.contains("double event_term_v = 0.0"));
        assert!(expanded.contains("if ((in.event > 0.0))"));
        assert!(expanded.contains("d_log(adjusted_v"));
        assert!(!expanded.contains("J2"));
        assert!(!expanded.contains("j2_"));
    }

    #[test]
    fn rejects_primary_dependent_runtime_branch() {
        let input = syn::parse2::<Input>(quote! {
            fn bad(q; event)
            leaves { log => log_stack => d_log }
            witnesses [];
            {
                let mut out = zero();
                if (q > 0.0) { out = compose(log, q); }
                return out;
            }
        })
        .expect("parse row program");
        let error = expand(input).expect_err("primary branch must be rejected");
        assert!(error.to_string().contains("unknown row_program scalar `q`"));
    }

    #[test]
    fn cuda_formulas_pin_sparse_mul_compose_and_mutable_support_union() {
        let cuda = emitted_cuda(quote! {
            fn formulas(x, y; take)
            leaves { curve => curve_stack => d_curve }
            witnesses [];
            {
                let product = mul(x, y);
                let curved = compose(curve, product);
                let mut out = x;
                if (take > 0.0) { out = add(curved, y); }
                return out;
            }
        });

        for formula in [
            "double product_g0 = y;",
            "double product_h0_1 = 1.0;",
            "double product_g1 = x;",
            "double curved_g0 = (product_g0 * curved_stack0[1]);",
            "double curved_h0_0 = (curved_stack0[2] * (product_g0 * product_g0));",
            "double curved_h0_1 = ((product_h0_1 * curved_stack0[1]) + (curved_stack0[2] * (product_g0 * product_g1)));",
            "double curved_g1 = (product_g1 * curved_stack0[1]);",
            "double curved_h1_1 = (curved_stack0[2] * (product_g1 * product_g1));",
            "double out_g0 = 1.0;",
            "double out_h0_0 = 0.0;",
            "double out_h0_1 = 0.0;",
            "double out_g1 = 0.0;",
            "double out_h1_1 = 0.0;",
            "out_g0 = curved_g0;",
            "out_h0_0 = curved_h0_0;",
            "out_h0_1 = curved_h0_1;",
            "out_g1 = (curved_g1 + 1.0);",
            "out_h1_1 = curved_h1_1;",
            "row_hessian[0] = out_h0_0;",
            "row_hessian[1] = out_h0_1;",
            "row_hessian[2] = out_h0_1;",
            "row_hessian[3] = out_h1_1;",
        ] {
            assert!(
                cuda.contains(formula),
                "missing generated formula: {formula}"
            );
        }
        assert!(!cuda.contains("* 0.0"));
        assert!(!cuda.contains("0.0 *"));
    }

    #[test]
    fn cuda_compose_temporaries_are_unique_across_repeated_assignments() {
        let cuda = emitted_cuda(quote! {
            fn repeated(q; event)
            leaves { log => log_stack => d_log }
            witnesses [];
            {
                let mut out = q;
                if (event > 0.0) {
                    out = compose(log, out);
                    out = compose(log, out);
                }
                return out;
            }
        });

        assert_eq!(cuda.matches("double out_stack0[3]").count(), 1);
        assert_eq!(cuda.matches("double out_stack1[3]").count(), 1);
        assert_eq!(cuda.matches("d_log(out_v, out_stack").count(), 2);
    }

    #[test]
    fn scalar_witness_schedule_is_dependency_sliced_from_the_same_program() {
        let witness = emitted_function(
            quote! {
                fn sliced(q, g; event)
                leaves {
                    sqrt => sqrt_stack => d_sqrt,
                    log => log_stack => d_log,
                }
                witnesses [adjusted];
                {
                    let square = add_constant(mul(g, g), 1.0);
                    let correction = compose(sqrt, square);
                    let adjusted = mul(q, correction);
                    let discarded = compose(log, adjusted);
                    return add(adjusted, discarded);
                }
            },
            "sliced_witnesses",
        );

        assert!(witness.contains("sqrt_stack"));
        assert!(witness.contains("adjusted"));
        assert!(!witness.contains("log_stack"));
        assert!(!witness.contains("discarded"));
        assert!(!witness.contains("event : f64"));
    }

    #[test]
    fn scalar_witness_schedule_retains_needed_branch_condition() {
        let witness = emitted_function(
            quote! {
                fn branched(q; event, unused)
                leaves {}
                witnesses [out];
                {
                    let mut out = zero();
                    if (event > 0.0) { out = q; }
                    return out;
                }
            },
            "branched_witnesses",
        );

        assert!(witness.contains("q : f64"));
        assert!(witness.contains("event : f64"));
        assert!(!witness.contains("unused : f64"));
    }
}

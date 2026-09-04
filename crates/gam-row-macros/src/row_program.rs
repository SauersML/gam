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
    kind: LeafKind,
}

/// How a leaf produces the derivative stack a `compose` consumes.
enum LeafKind {
    /// A stack builder: one Rust function and one CUDA function, each taking
    /// the composition point followed by the compose's scalar arguments.
    Function { rust: Path, cuda: Ident },
    /// The stack is supplied: the compose's five scalar arguments ARE
    /// `[value, first, second, third, fourth]` at the composition point, as
    /// the kernel builder evaluated them there. Nothing is called and the
    /// point is not inspected. Before this kind existed such a stack went
    /// through a builder whose only use of the point was a NaN select, eleven
    /// instructions per composition in release code, guarding a state the
    /// kernel builders cannot construct (#932).
    Supplied,
}

impl Leaf {
    /// The Rust expression producing this leaf's stack at `point`.
    fn rust_application(&self, point: TokenStream2, arguments: &[Expr]) -> TokenStream2 {
        match &self.kind {
            LeafKind::Function { rust, .. } => quote!(#rust(#point, #(#arguments),*)),
            LeafKind::Supplied => quote!([#(#arguments),*]),
        }
    }

    /// [`Self::rust_application`] as source text for the string lowerings.
    fn rust_application_source(&self, point: &str, arguments: &[String]) -> String {
        match &self.kind {
            LeafKind::Function { rust, .. } => {
                let rust = quote!(#rust).to_string();
                let mut all = vec![point.to_string()];
                all.extend(arguments.iter().cloned());
                format!("{rust}({})", all.join(", "))
            }
            LeafKind::Supplied => format!("[{}]", arguments.join(", ")),
        }
    }

    /// The CUDA prelude declaring `stack` (three entries: the order-2
    /// lowering reads value, first and second) and filling it.
    fn cuda_stack_prelude(&self, point: &str, arguments: &[String], stack: &str) -> String {
        match &self.kind {
            LeafKind::Function { cuda, .. } => {
                let mut all = vec![point.to_string()];
                all.extend(arguments.iter().cloned());
                all.push(stack.to_string());
                format!("double {stack}[3];\n{cuda}({});", all.join(", "))
            }
            LeafKind::Supplied => {
                format!("double {stack}[3] = {{{}}};", arguments[..3].join(", "))
            }
        }
    }
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

#[derive(Default)]
struct EmissionSurfaces {
    generic: bool,
    runtime: bool,
    order2: bool,
    third: bool,
    fourth: bool,
    full: bool,
    witnesses: bool,
    cuda: bool,
}

impl EmissionSurfaces {
    fn insert(&mut self, surface: &Ident) -> Result<()> {
        let selected = match surface.to_string().as_str() {
            "generic" => &mut self.generic,
            "runtime" => &mut self.runtime,
            "order2" => &mut self.order2,
            "third" => &mut self.third,
            "fourth" => &mut self.fourth,
            "full" => &mut self.full,
            "witnesses" => &mut self.witnesses,
            "cuda" => &mut self.cuda,
            _ => {
                return Err(syn::Error::new_spanned(
                    surface,
                    "row_program emission surface must be one of `generic`, `runtime`, `order2`, `third`, `fourth`, `full`, `witnesses`, or `cuda`",
                ));
            }
        };
        if *selected {
            return Err(syn::Error::new_spanned(
                surface,
                format!("duplicate row_program emission surface `{surface}`"),
            ));
        }
        *selected = true;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        !(self.generic
            || self.runtime
            || self.order2
            || self.third
            || self.fourth
            || self.full
            || self.witnesses
            || self.cuda)
    }
}

pub(crate) struct Input {
    visibility: Visibility,
    name: Ident,
    primaries: Vec<Ident>,
    constants: Vec<Ident>,
    /// Constants declared `name: sign`: a value in `{-1, +1}`, so its square
    /// is one and a composition on `scale(x, s)` reads `f''` unscaled.
    signs: Vec<Ident>,
    emissions: EmissionSurfaces,
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
        let mut signs = Vec::new();
        if arguments.peek(Token![;]) {
            arguments.parse::<Token![;]>()?;
            while !arguments.is_empty() {
                let constant = arguments.parse::<Ident>()?;
                if arguments.peek(Token![:]) {
                    arguments.parse::<Token![:]>()?;
                    let role = arguments.parse::<Ident>()?;
                    if role != "sign" {
                        return Err(syn::Error::new_spanned(
                            role,
                            "row_program constant roles: `sign` (a value in {-1, +1})",
                        ));
                    }
                    signs.push(constant.clone());
                }
                constants.push(constant);
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

        let emit_keyword = input.parse::<Ident>()?;
        if emit_keyword != "emit" {
            return Err(syn::Error::new_spanned(
                emit_keyword,
                "row_program expects mandatory `emit [ ... ];` surfaces",
            ));
        }
        let emission_tokens;
        bracketed!(emission_tokens in input);
        let mut emissions = EmissionSurfaces::default();
        while !emission_tokens.is_empty() {
            let surface = emission_tokens.parse::<Ident>()?;
            emissions.insert(&surface)?;
            if emission_tokens.peek(Token![,]) {
                emission_tokens.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        if emissions.is_empty() {
            return Err(emission_tokens.error("row_program must emit at least one surface"));
        }
        if !emission_tokens.is_empty() {
            return Err(emission_tokens.error("invalid row_program emission surface list"));
        }
        input.parse::<Token![;]>()?;

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
            let rust: Path = leaf_tokens.parse()?;
            let kind = if rust.is_ident("supplied") && !leaf_tokens.peek(Token![=>]) {
                LeafKind::Supplied
            } else {
                leaf_tokens.parse::<Token![=>]>()?;
                LeafKind::Function {
                    rust,
                    cuda: leaf_tokens.parse()?,
                }
            };
            leaves.push(Leaf { alias, kind });
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
            signs,
            emissions,
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

/// The bare identifier a call expression invokes, or a spanned error naming
/// `what` (the construct that requires a bare name) when the callee is a path
/// or anything else.
pub(crate) fn bare_call_name<'a>(call: &'a ExprCall, what: &str) -> Result<&'a Ident> {
    let message = format!("{what} must use a bare function name");
    let Expr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new_spanned(&call.func, message));
    };
    path.path
        .get_ident()
        .ok_or_else(|| syn::Error::new_spanned(&call.func, message))
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
                | BinOp::And(_)
                | BinOp::Or(_)
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
            let operation = bare_call_name(call, "row_program operations")?.to_string();
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
            let stack = leaves[*leaf].rust_application(quote!(value.value()), arguments);
            quote!({
                let value = #value_ident;
                value.compose_unary(#stack)
            })
        }
    }
}

fn rust_runtime_expression(expression: &ProgramExpr, leaves: &[Leaf]) -> TokenStream2 {
    match expression {
        ProgramExpr::Path(ident) => quote!(#ident.clone()),
        ProgramExpr::Zero => quote!(S::constant(
            0.0,
            __row_program_dimension,
            __row_program_workspace
        )),
        ProgramExpr::Neg(value) => {
            let value = rust_runtime_expression(value, leaves);
            quote!({ let value = #value; value.neg() })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = rust_runtime_expression(value, leaves);
            quote!({ let value = #value; value.scale(#scalar) })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let value = rust_runtime_expression(value, leaves);
            quote!({
                let value = #value;
                value.add_constant(#scalar)
            })
        }
        ProgramExpr::Add(left, right) => {
            let left = rust_runtime_expression(left, leaves);
            let right = rust_runtime_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.add(&right) })
        }
        ProgramExpr::Mul(left, right) => {
            let left = rust_runtime_expression(left, leaves);
            let right = rust_runtime_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.mul(&right) })
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let value_ident = value;
            let stack = leaves[*leaf].rust_application(quote!(value.value()), arguments);
            quote!({
                let value = #value_ident.clone();
                value.compose_unary(#stack)
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
            let stack = leaves[*leaf].rust_application(quote!(#value), arguments);
            quote!(#stack[0])
        }
    }
}

/// The primaries as parameters of one direct lowering. A primary whose VALUE
/// the body never reads -- it composes a supplied stack and enters only as a
/// differentiation axis -- keeps its position under an underscore name, so
/// every caller stays positional and the generated function is lint-clean
/// under `deny(warnings)`.
fn primary_parameters(primaries: &[Ident], body: &TokenStream2) -> Vec<Ident> {
    let text = body.to_string();
    primaries
        .iter()
        .map(|primary| {
            let name = primary.to_string();
            let used = text
                .split(|character: char| !(character.is_alphanumeric() || character == '_'))
                .any(|token| token == name);
            if used {
                primary.clone()
            } else {
                format_ident!("_{name}")
            }
        })
        .collect()
}

/// A supplied stack composes exactly its five entries; a builder leaf takes
/// whatever scalar arguments its function declares.
fn validate_supplied_stacks(expression: &ProgramExpr, leaves: &[Leaf]) -> Result<()> {
    match expression {
        ProgramExpr::Path(..) | ProgramExpr::Zero => Ok(()),
        ProgramExpr::Neg(value)
        | ProgramExpr::Scale(value, ..)
        | ProgramExpr::AddConstant(value, ..) => validate_supplied_stacks(value, leaves),
        ProgramExpr::Add(left, right) | ProgramExpr::Mul(left, right) => {
            validate_supplied_stacks(left, leaves)?;
            validate_supplied_stacks(right, leaves)
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            if matches!(leaves[*leaf].kind, LeafKind::Supplied) && arguments.len() != 5 {
                return Err(syn::Error::new_spanned(
                    value,
                    format!(
                        "a supplied stack composes exactly five entries `[value, first, second, third, fourth]`, got {}",
                        arguments.len()
                    ),
                ));
            }
            Ok(())
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

fn collect_scalar_expression_dependencies(
    expression: &Expr,
    dependencies: &mut HashSet<String>,
) -> Result<()> {
    match expression {
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                dependencies.insert(ident.to_string());
            }
            Ok(())
        }
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            collect_scalar_expression_dependencies(expr, dependencies)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => collect_scalar_expression_dependencies(expr, dependencies),
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
            collect_scalar_expression_dependencies(left, dependencies)?;
            collect_scalar_expression_dependencies(right, dependencies)
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(()),
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_program scalar dependency expression",
        )),
    }
}

fn collect_program_scalar_dependencies(
    expression: &ProgramExpr,
    dependencies: &mut HashSet<String>,
) -> Result<()> {
    match expression {
        ProgramExpr::Path(_) | ProgramExpr::Zero => Ok(()),
        ProgramExpr::Neg(value) => collect_program_scalar_dependencies(value, dependencies),
        ProgramExpr::Scale(value, scalar) | ProgramExpr::AddConstant(value, scalar) => {
            collect_program_scalar_dependencies(value, dependencies)?;
            collect_scalar_expression_dependencies(scalar, dependencies)
        }
        ProgramExpr::Add(left, right) | ProgramExpr::Mul(left, right) => {
            collect_program_scalar_dependencies(left, dependencies)?;
            collect_program_scalar_dependencies(right, dependencies)
        }
        ProgramExpr::Compose { arguments, .. } => {
            for argument in arguments {
                collect_scalar_expression_dependencies(argument, dependencies)?;
            }
            Ok(())
        }
    }
}

fn witness_scalar_dependencies(
    statements: &[Statement],
    jet_dependencies: &HashSet<String>,
) -> Result<HashSet<String>> {
    let mut dependencies = HashSet::new();
    for statement in statements {
        match statement {
            Statement::Local { name, value, .. }
                if jet_dependencies.contains(&name.to_string()) =>
            {
                collect_program_scalar_dependencies(value, &mut dependencies)?;
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut condition_is_needed = false;
                for (target, value) in assignments {
                    if jet_dependencies.contains(&target.to_string()) {
                        collect_program_scalar_dependencies(value, &mut dependencies)?;
                        condition_is_needed = true;
                    }
                }
                if condition_is_needed {
                    collect_scalar_expression_dependencies(condition, &mut dependencies)?;
                }
            }
            Statement::Local { .. } => {}
        }
    }
    Ok(dependencies)
}

#[derive(Clone, Copy)]
enum SymbolicTarget {
    Rust,
    Cuda,
}

fn symbolic_scalar(
    expression: &Expr,
    constants: &HashSet<String>,
    target: SymbolicTarget,
) -> Result<String> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if constants.contains(&ident.to_string()) {
                Ok(match target {
                    SymbolicTarget::Rust => ident.to_string(),
                    SymbolicTarget::Cuda => format!("in.{ident}"),
                })
            } else {
                Err(syn::Error::new_spanned(
                    ident,
                    "unknown row_program symbolic scalar",
                ))
            }
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(quote!(#literal).to_string()),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            Ok(format!("({})", symbolic_scalar(expr, constants, target)?))
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => Ok(format!("-({})", symbolic_scalar(expr, constants, target)?)),
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
                BinOp::And(_) => "&&",
                BinOp::Or(_) => "||",
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "unsupported row_program symbolic scalar operator",
                    ));
                }
            };
            Ok(format!(
                "({} {operator} {})",
                symbolic_scalar(left, constants, target)?,
                symbolic_scalar(right, constants, target)?
            ))
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_program symbolic scalar expression",
        )),
    }
}

#[derive(Clone)]
struct SymbolicJet {
    value: String,
    gradient: Vec<Option<String>>,
    // Only entries with a <= b are populated. The generated CUDA computes the
    // packed triangle once and scatters it symmetrically at the output seam.
    hessian: Vec<Option<String>>,
}

#[derive(Clone)]
struct SymbolicSupport {
    gradient: Vec<bool>,
    hessian: Vec<bool>,
}

impl SymbolicSupport {
    fn empty(dimension: usize) -> Self {
        Self {
            gradient: vec![false; dimension],
            hessian: vec![false; dimension * dimension],
        }
    }

    fn include(&mut self, jet: &SymbolicJet) {
        for (present, component) in self.gradient.iter_mut().zip(&jet.gradient) {
            *present |= component.is_some();
        }
        for (present, component) in self.hessian.iter_mut().zip(&jet.hessian) {
            *present |= component.is_some();
        }
    }
}

impl SymbolicJet {
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

    fn constant(value: String, dimension: usize) -> Self {
        let mut out = Self::zero(dimension);
        out.value = value;
        out
    }

    fn support(&self) -> SymbolicSupport {
        let mut support = SymbolicSupport::empty(self.gradient.len());
        support.include(self);
        support
    }

    fn reference(name: &str, support: &SymbolicSupport, dimension: usize) -> Self {
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

fn symbolic_is_zero(value: &str) -> bool {
    value == "0.0"
}

fn symbolic_is_one(value: &str) -> bool {
    value == "1.0"
}

fn symbolic_is_negative_one(value: &str) -> bool {
    matches!(value, "-1.0" | "-(1.0)" | "(-1.0)")
}

fn symbolic_negate(value: &str) -> String {
    if symbolic_is_zero(value) {
        "0.0".to_string()
    } else if symbolic_is_negative_one(value) {
        "1.0".to_string()
    } else if symbolic_is_one(value) {
        "-1.0".to_string()
    } else {
        format!("-({value})")
    }
}

fn symbolic_add(left: &str, right: &str) -> String {
    if symbolic_is_zero(left) {
        right.to_string()
    } else if symbolic_is_zero(right) {
        left.to_string()
    } else {
        format!("({left} + {right})")
    }
}

fn symbolic_multiply(left: &str, right: &str) -> String {
    if symbolic_is_zero(left) || symbolic_is_zero(right) {
        "0.0".to_string()
    } else if symbolic_is_one(left) {
        right.to_string()
    } else if symbolic_is_one(right) {
        left.to_string()
    } else if symbolic_is_negative_one(left) {
        symbolic_negate(right)
    } else if symbolic_is_negative_one(right) {
        symbolic_negate(left)
    } else {
        format!("({left} * {right})")
    }
}

fn symbolic_add_component(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(symbolic_add(left, right)),
        (Some(value), None) | (None, Some(value)) => Some(value.clone()),
        (None, None) => None,
    }
}

fn symbolic_multiply_component(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(symbolic_multiply(left, right)),
        _ => None,
    }
}

fn symbolic_scale_component(component: &Option<String>, scalar: &str) -> Option<String> {
    component
        .as_ref()
        .map(|component| symbolic_multiply(component, scalar))
}

fn symbolic_add_jets(left: SymbolicJet, right: SymbolicJet) -> SymbolicJet {
    SymbolicJet {
        value: symbolic_add(&left.value, &right.value),
        gradient: left
            .gradient
            .iter()
            .zip(&right.gradient)
            .map(|(left, right)| symbolic_add_component(left, right))
            .collect(),
        hessian: left
            .hessian
            .iter()
            .zip(&right.hessian)
            .map(|(left, right)| symbolic_add_component(left, right))
            .collect(),
    }
}

fn symbolic_multiply_jets(left: SymbolicJet, right: SymbolicJet) -> SymbolicJet {
    let dimension = left.gradient.len();
    let mut gradient = vec![None; dimension];
    let mut hessian = vec![None; dimension * dimension];
    for axis in 0..dimension {
        gradient[axis] = symbolic_add_component(
            &symbolic_scale_component(&right.gradient[axis], &left.value),
            &symbolic_scale_component(&left.gradient[axis], &right.value),
        );
        for other in axis..dimension {
            let index = axis * dimension + other;
            let inherited_right = symbolic_scale_component(&right.hessian[index], &left.value);
            let cross_forward =
                symbolic_multiply_component(&left.gradient[axis], &right.gradient[other]);
            let cross_reverse =
                symbolic_multiply_component(&left.gradient[other], &right.gradient[axis]);
            let inherited_left = symbolic_scale_component(&left.hessian[index], &right.value);
            hessian[index] = symbolic_add_component(
                &symbolic_add_component(
                    &symbolic_add_component(&inherited_right, &cross_forward),
                    &cross_reverse,
                ),
                &inherited_left,
            );
        }
    }
    SymbolicJet {
        value: symbolic_multiply(&left.value, &right.value),
        gradient,
        hessian,
    }
}

fn symbolic_negate_jet(value: SymbolicJet) -> SymbolicJet {
    SymbolicJet {
        value: symbolic_negate(&value.value),
        gradient: value
            .gradient
            .iter()
            .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
            .collect(),
        hessian: value
            .hessian
            .iter()
            .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
            .collect(),
    }
}

fn symbolic_scale_jet(value: SymbolicJet, scalar: &str) -> SymbolicJet {
    SymbolicJet {
        value: symbolic_multiply(&value.value, scalar),
        gradient: value
            .gradient
            .iter()
            .map(|component| symbolic_scale_component(component, scalar))
            .collect(),
        hessian: value
            .hessian
            .iter()
            .map(|component| symbolic_scale_component(component, scalar))
            .collect(),
    }
}

fn symbolic_compose_jet(input: SymbolicJet, stack: &str, offset: usize) -> SymbolicJet {
    let first = format!("{stack}[{}]", offset + 1);
    let second = format!("{stack}[{}]", offset + 2);
    let dimension = input.gradient.len();
    let mut gradient = vec![None; dimension];
    let mut hessian = vec![None; dimension * dimension];
    for axis in 0..dimension {
        gradient[axis] = symbolic_scale_component(&input.gradient[axis], &first);
        for other in axis..dimension {
            let index = axis * dimension + other;
            let inherited = symbolic_scale_component(&input.hessian[index], &first);
            let curvature =
                symbolic_multiply_component(&input.gradient[axis], &input.gradient[other])
                    .map(|component| symbolic_multiply(&second, &component));
            hessian[index] = symbolic_add_component(&inherited, &curvature);
        }
    }
    SymbolicJet {
        value: format!("{stack}[{offset}]"),
        gradient,
        hessian,
    }
}

/// A local that is exactly a scalar multiple (or the negation) of another
/// binding: `let m = scale(x, s);` or `let m = neg(x);`.
///
/// A composition on such a local pulls the scalar into the outer derivative
/// stack instead of scaling every channel of the point: for `f(s·x)`,
/// `∂_a = (s·f')·x_a` and `∂_ab = (s²·f'')·x_a·x_b + (s·f')·x_ab`, so the
/// two scaled stack entries are formed once and the chain rule reads `x`'s
/// own channels. The rigid Bernoulli row composes its probit leaf on
/// `scale(latent_index, outcome_sign)`: scaling the index's six channels by
/// the sign cost six multiplies per row where the hand kernel pays two
/// (`m = s·η`, `u1 = s·k1`), which was the whole margin by which the hand
/// won (#932). The scaled local itself is still emitted; whatever of it
/// nothing reads (its scaled derivative channels, when only the compose
/// consumed them) is dropped by the scheduler.
struct ScaledAlias {
    inner: String,
    /// `None` is a negation.
    scalar: Option<Expr>,
}

fn symbolic_expression(
    expression: &ProgramExpr,
    owner: &str,
    leaves: &[Leaf],
    constants: &HashSet<String>,
    signs: &HashSet<String>,
    bindings: &HashMap<String, SymbolicJet>,
    aliases: &HashMap<String, ScaledAlias>,
    target: SymbolicTarget,
    dimension: usize,
    stack_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<SymbolicJet> {
    let mut child = |expression: &ProgramExpr| {
        symbolic_expression(
            expression,
            owner,
            leaves,
            constants,
            signs,
            bindings,
            aliases,
            target,
            dimension,
            stack_index,
            preludes,
        )
    };
    match expression {
        ProgramExpr::Path(ident) => bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
            syn::Error::new_spanned(ident, "symbolic row_program binding is not defined")
        }),
        ProgramExpr::Zero => Ok(SymbolicJet::zero(dimension)),
        ProgramExpr::Neg(value) => {
            let value = child(value)?;
            Ok(SymbolicJet {
                value: symbolic_negate(&value.value),
                gradient: value
                    .gradient
                    .iter()
                    .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
                    .collect(),
                hessian: value
                    .hessian
                    .iter()
                    .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
                    .collect(),
            })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = child(value)?;
            let scalar = symbolic_scalar(scalar, constants, target)?;
            Ok(SymbolicJet {
                value: symbolic_multiply(&value.value, &scalar),
                gradient: value
                    .gradient
                    .iter()
                    .map(|component| symbolic_scale_component(component, &scalar))
                    .collect(),
                hessian: value
                    .hessian
                    .iter()
                    .map(|component| symbolic_scale_component(component, &scalar))
                    .collect(),
            })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            value.value = symbolic_add(&value.value, &symbolic_scalar(scalar, constants, target)?);
            Ok(value)
        }
        ProgramExpr::Add(left, right) => Ok(symbolic_add_jets(child(left)?, child(right)?)),
        ProgramExpr::Mul(left, right) => Ok(symbolic_multiply_jets(child(left)?, child(right)?)),
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "symbolic compose input is not defined")
            })?;
            let suffix = *stack_index;
            *stack_index += 1;
            let stack = format!("{owner}_stack{suffix}");
            let mut leaf_arguments = Vec::new();
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, target)?);
            }
            match target {
                SymbolicTarget::Rust => {
                    let application =
                        leaves[*leaf].rust_application_source(&input.value, &leaf_arguments);
                    preludes.push(format!("let {stack} = {application};"));
                }
                SymbolicTarget::Cuda => {
                    preludes.push(leaves[*leaf].cuda_stack_prelude(
                        &input.value,
                        &leaf_arguments,
                        &stack,
                    ));
                }
            }

            // A scaled composition point: the scalar goes into the outer
            // stack once, and the chain rule reads the unscaled binding.
            let (input, first, second) = match aliases
                .get(&value.to_string())
                .and_then(|alias| bindings.get(&alias.inner).map(|inner| (alias, inner)))
            {
                Some((alias, inner)) => {
                    let first = format!("{stack}_u1");
                    let second = format!("{stack}_u2");
                    let (first_value, second_value) = match &alias.scalar {
                        Some(scalar) => {
                            let is_sign = match scalar {
                                Expr::Path(path) => path_ident(path)
                                    .is_ok_and(|ident| signs.contains(&ident.to_string())),
                                _ => false,
                            };
                            let scalar = symbolic_scalar(scalar, constants, target)?;
                            (
                                symbolic_multiply(&format!("{stack}[1]"), &scalar),
                                // A sign squares to one: `f''` is read as is.
                                if is_sign {
                                    format!("{stack}[2]")
                                } else {
                                    symbolic_multiply(
                                        &symbolic_multiply(&format!("{stack}[2]"), &scalar),
                                        &scalar,
                                    )
                                },
                            )
                        }
                        None => (symbolic_negate(&format!("{stack}[1]")), format!("{stack}[2]")),
                    };
                    match target {
                        SymbolicTarget::Rust => {
                            preludes.push(format!("let {first}: f64 = {first_value};"));
                            preludes.push(format!("let {second}: f64 = {second_value};"));
                        }
                        SymbolicTarget::Cuda => {
                            preludes.push(format!("double {first} = {first_value};"));
                            preludes.push(format!("double {second} = {second_value};"));
                        }
                    }
                    (inner.clone(), first, second)
                }
                None => (input, format!("{stack}[1]"), format!("{stack}[2]")),
            };
            // The curvature term of the chain rule, `f''·g_a·g_b`, is emitted as
            // `(f''·g_a)·g_b` with `f''·g_a` hoisted ONCE per present axis, so a
            // Hessian entry costs one multiply instead of two. LLVM cannot do
            // this itself: without fast-math it may not re-associate
            // `f''·(g_a·g_b)`. Over a support of `s` axes this removes
            // `s(s-1)/2` multiplies per composition; a hand kernel that writes
            // `second * g_a * g_b` per entry carries them all.
            let mut curvature_coefficient: Vec<Option<String>> = vec![None; dimension];
            for axis in 0..dimension {
                if let Some(component) = &input.gradient[axis] {
                    let product = symbolic_multiply(&second, component);
                    let name = format!("{stack}_c{axis}");
                    match target {
                        SymbolicTarget::Rust => {
                            preludes.push(format!("let {name}: f64 = {product};"));
                        }
                        SymbolicTarget::Cuda => {
                            preludes.push(format!("double {name} = {product};"));
                        }
                    }
                    curvature_coefficient[axis] = Some(name);
                }
            }
            let mut gradient = vec![None; dimension];
            let mut hessian = vec![None; dimension * dimension];
            for axis in 0..dimension {
                gradient[axis] = symbolic_scale_component(&input.gradient[axis], &first);
                for other in axis..dimension {
                    let index = axis * dimension + other;
                    let inherited = symbolic_scale_component(&input.hessian[index], &first);
                    let curvature = symbolic_multiply_component(
                        &curvature_coefficient[axis],
                        &input.gradient[other],
                    );
                    hessian[index] = symbolic_add_component(&inherited, &curvature);
                }
            }
            Ok(SymbolicJet {
                value: format!("{stack}[0]"),
                gradient,
                hessian,
            })
        }
    }
}

#[derive(Clone)]
struct DirectionalJet {
    base: SymbolicJet,
    u: SymbolicJet,
    v: SymbolicJet,
    uv: SymbolicJet,
}

#[derive(Clone)]
struct DirectionalSupport {
    base: SymbolicSupport,
    u: SymbolicSupport,
    v: SymbolicSupport,
    uv: SymbolicSupport,
}

impl DirectionalSupport {
    fn empty(dimension: usize) -> Self {
        Self {
            base: SymbolicSupport::empty(dimension),
            u: SymbolicSupport::empty(dimension),
            v: SymbolicSupport::empty(dimension),
            uv: SymbolicSupport::empty(dimension),
        }
    }

    fn include(&mut self, jet: &DirectionalJet) {
        self.base.include(&jet.base);
        self.u.include(&jet.u);
        self.v.include(&jet.v);
        self.uv.include(&jet.uv);
    }
}

impl DirectionalJet {
    fn zero(dimension: usize) -> Self {
        Self {
            base: SymbolicJet::zero(dimension),
            u: SymbolicJet::zero(dimension),
            v: SymbolicJet::zero(dimension),
            uv: SymbolicJet::zero(dimension),
        }
    }

    fn primary(name: &str, axis: usize, dimension: usize, fourth: bool) -> Self {
        Self {
            base: SymbolicJet::primary(name, axis, dimension),
            u: SymbolicJet::constant(format!("direction_u[{axis}]"), dimension),
            v: if fourth {
                SymbolicJet::constant(format!("direction_v[{axis}]"), dimension)
            } else {
                SymbolicJet::zero(dimension)
            },
            uv: SymbolicJet::zero(dimension),
        }
    }

    fn support(&self) -> DirectionalSupport {
        let mut support = DirectionalSupport::empty(self.base.gradient.len());
        support.include(self);
        support
    }
}

fn directional_add(left: DirectionalJet, right: DirectionalJet) -> DirectionalJet {
    DirectionalJet {
        base: symbolic_add_jets(left.base, right.base),
        u: symbolic_add_jets(left.u, right.u),
        v: symbolic_add_jets(left.v, right.v),
        uv: symbolic_add_jets(left.uv, right.uv),
    }
}

fn directional_negate(value: DirectionalJet) -> DirectionalJet {
    DirectionalJet {
        base: symbolic_negate_jet(value.base),
        u: symbolic_negate_jet(value.u),
        v: symbolic_negate_jet(value.v),
        uv: symbolic_negate_jet(value.uv),
    }
}

fn directional_scale(value: DirectionalJet, scalar: &str) -> DirectionalJet {
    DirectionalJet {
        base: symbolic_scale_jet(value.base, scalar),
        u: symbolic_scale_jet(value.u, scalar),
        v: symbolic_scale_jet(value.v, scalar),
        uv: symbolic_scale_jet(value.uv, scalar),
    }
}

fn directional_multiply(
    left: DirectionalJet,
    right: DirectionalJet,
    fourth: bool,
) -> DirectionalJet {
    let base = symbolic_multiply_jets(left.base.clone(), right.base.clone());
    let u = symbolic_add_jets(
        symbolic_multiply_jets(left.u.clone(), right.base.clone()),
        symbolic_multiply_jets(left.base.clone(), right.u.clone()),
    );
    if !fourth {
        return DirectionalJet {
            base,
            u,
            v: SymbolicJet::zero(left.base.gradient.len()),
            uv: SymbolicJet::zero(left.base.gradient.len()),
        };
    }
    let v = symbolic_add_jets(
        symbolic_multiply_jets(left.v.clone(), right.base.clone()),
        symbolic_multiply_jets(left.base.clone(), right.v.clone()),
    );
    let uv = symbolic_add_jets(
        symbolic_add_jets(
            symbolic_multiply_jets(left.uv, right.base.clone()),
            symbolic_multiply_jets(left.u, right.v),
        ),
        symbolic_add_jets(
            symbolic_multiply_jets(left.v, right.u),
            symbolic_multiply_jets(left.base, right.uv),
        ),
    );
    DirectionalJet { base, u, v, uv }
}

fn materialize_directional(
    value: DirectionalJet,
    owner: &str,
    fourth: bool,
    temporary_index: &mut usize,
    preludes: &mut Vec<String>,
) -> DirectionalJet {
    let name = format!("{owner}_directional_tmp{}", *temporary_index);
    *temporary_index += 1;
    let support = value.support();
    let mut source = String::new();
    push_directional_declaration(&mut source, "", &name, "", &value, &support, fourth);
    preludes.push(source);
    directional_reference(&name, &support, value.base.gradient.len(), fourth)
}

struct DirectionalExpressionEnvironment<'a> {
    leaves: &'a [Leaf],
    constants: &'a HashSet<String>,
    dimension: usize,
    fourth: bool,
}

/// An exact rational carried beside a dense Taylor coefficient's emitted
/// expression: the coefficient's value is `factor · expression`, and the
/// factor never reaches the row until a consumer needs the value.
///
/// A composition on a primary introduces `1/k!` per coefficient and a
/// derivative extraction removes it with `k!`; emitted as two multiplies
/// they cost the row two multiplies per coefficient (LLVM folds
/// `(x · 0.5) · 2` only under fast-math). The rigid Bernoulli row's fourth
/// channel paid seventeen of them, which was the whole margin by which the
/// hand kernel won on one host (#932). With the factor kept exact in the
/// emitter, products multiply factors, a sum keeps the most common factor
/// of its terms out of the arithmetic, and the residual at extraction is
/// one for every coefficient of a jet built from compositions on primaries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };

    fn new(numerator: i64, denominator: i64) -> Self {
        assert!(denominator != 0, "a dense Taylor factor has a nonzero denominator");
        fn gcd(left: i64, right: i64) -> i64 {
            if right == 0 {
                left.abs()
            } else {
                gcd(right, left % right)
            }
        }
        let divisor = gcd(numerator, denominator).max(1);
        let sign = if denominator < 0 { -1 } else { 1 };
        Self {
            numerator: sign * numerator / divisor,
            denominator: sign * denominator / divisor,
        }
    }

    fn factorial(count: usize) -> Self {
        Self::new((2..=count as i64).product(), 1)
    }

    fn times(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )
    }

    fn over(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.denominator,
            self.denominator * other.numerator,
        )
    }

    fn is_one(self) -> bool {
        self == Self::ONE
    }

    fn literal(self) -> String {
        if self.denominator == 1 {
            format!("{}.0", self.numerator)
        } else {
            format!("{:.17}", self.numerator as f64 / self.denominator as f64)
        }
    }
}

/// `expression · factor`, with no multiply when the factor is one.
fn scaled_by(expression: &str, factor: Rational) -> String {
    if factor.is_one() {
        expression.to_string()
    } else {
        symbolic_multiply(expression, &factor.literal())
    }
}

/// The factor that occurs most often among `factors` (the earliest on a
/// tie), or `None` when there are none.
fn most_common_factor(factors: impl Iterator<Item = Rational>) -> Option<Rational> {
    let factors: Vec<Rational> = factors.collect();
    let mut common: Option<Rational> = None;
    let mut best = 0;
    for candidate in &factors {
        let count = factors.iter().filter(|factor| *factor == candidate).count();
        if count > best {
            best = count;
            common = Some(*candidate);
        }
    }
    common
}

/// Sum terms `factor · expression` into one `factor · expression`: the most
/// common factor among the terms is kept out of the arithmetic, and each
/// term with another factor pays one multiply by the ratio. The terms keep
/// their order.
fn combine_dense_taylor_terms(terms: &[(Rational, String)]) -> Option<(Rational, String)> {
    let common = most_common_factor(terms.iter().map(|(factor, _)| *factor))?;
    let mut sum: Option<String> = None;
    for (factor, expression) in terms {
        let term = scaled_by(expression, factor.over(common));
        sum = symbolic_add_component(&sum, &Some(term));
    }
    sum.map(|sum| (common, sum))
}

/// Exact normalized multivariate Taylor coefficients through degree four.
///
/// The directional lowering is asymptotically right for wide rows because it
/// never materializes a dense high-order tensor. For one- and two-primary row
/// programs, however, propagating four second-order directional jets performs
/// more arithmetic than propagating the complete tiny Taylor polynomial once.
/// This representation is a compile-time algebra only: emitted production code
/// contains direct scalar formulas, not an automatic-differentiation runtime.
///
/// Each coefficient is `factor · expression` (see [`Rational`]).
#[derive(Clone)]
struct DenseTaylorJet {
    dimension: usize,
    order: usize,
    coefficients: Vec<Option<String>>,
    factors: Vec<Rational>,
}

fn dense_taylor_slot_count(dimension: usize) -> usize {
    5usize.pow(dimension as u32)
}

fn dense_taylor_counts(mut index: usize, dimension: usize) -> Vec<usize> {
    let mut counts = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        counts.push(index % 5);
        index /= 5;
    }
    counts
}

fn dense_taylor_index(counts: &[usize]) -> usize {
    counts
        .iter()
        .rev()
        .fold(0usize, |index, count| index * 5 + count)
}

fn dense_taylor_component(value: String, index: usize) -> Option<String> {
    if index != 0 && symbolic_is_zero(&value) {
        None
    } else {
        Some(value)
    }
}

impl DenseTaylorJet {
    fn zero(dimension: usize, order: usize) -> Self {
        let slots = dense_taylor_slot_count(dimension);
        let mut coefficients = vec![None; slots];
        coefficients[0] = Some("0.0".to_string());
        Self {
            dimension,
            order,
            coefficients,
            factors: vec![Rational::ONE; slots],
        }
    }

    fn constant(value: String, dimension: usize, order: usize) -> Self {
        let mut out = Self::zero(dimension, order);
        out.coefficients[0] = Some(value);
        out
    }

    fn primary(name: &str, axis: usize, dimension: usize, order: usize) -> Self {
        let mut out = Self::constant(name.to_string(), dimension, order);
        let mut counts = vec![0usize; dimension];
        counts[axis] = 1;
        out.coefficients[dense_taylor_index(&counts)] = Some("1.0".to_string());
        out
    }

    fn support(&self) -> Vec<bool> {
        self.coefficients.iter().map(Option::is_some).collect()
    }

    fn reference(
        name: &str,
        support: &[bool],
        factors: &[Rational],
        dimension: usize,
        order: usize,
    ) -> Self {
        let mut out = Self::zero(dimension, order);
        for (index, present) in support.iter().copied().enumerate() {
            if present {
                out.coefficients[index] = Some(format!("{name}_c{index}"));
                out.factors[index] = factors[index];
            }
        }
        out
    }

    /// The coefficient's value, its factor applied.
    fn value(&self, index: usize) -> Option<String> {
        self.coefficients[index]
            .as_ref()
            .map(|coefficient| scaled_by(coefficient, self.factors[index]))
    }

    /// Every factor applied into its expression. A mutable local is stored
    /// this way: its gates assign it values with factors of their own, and
    /// one name can carry only one.
    fn normalized(mut self) -> Self {
        for index in 0..self.coefficients.len() {
            self.coefficients[index] = self.value(index);
            self.factors[index] = Rational::ONE;
        }
        self
    }

    fn set(&mut self, index: usize, term: Option<(Rational, String)>) {
        let (factor, coefficient) = match term {
            Some((factor, expression)) => (factor, dense_taylor_component(expression, index)),
            None => (Rational::ONE, None),
        };
        self.factors[index] = if coefficient.is_some() {
            factor
        } else {
            Rational::ONE
        };
        self.coefficients[index] = coefficient;
    }
}

fn dense_taylor_add(left: DenseTaylorJet, right: DenseTaylorJet) -> DenseTaylorJet {
    let mut out = DenseTaylorJet::zero(left.dimension, left.order);
    for index in 0..out.coefficients.len() {
        let mut terms = Vec::new();
        if let Some(component) = &left.coefficients[index] {
            terms.push((left.factors[index], component.clone()));
        }
        if let Some(component) = &right.coefficients[index] {
            terms.push((right.factors[index], component.clone()));
        }
        out.set(index, combine_dense_taylor_terms(&terms));
    }
    if out.coefficients[0].is_none() {
        out.coefficients[0] = Some("0.0".to_string());
    }
    out
}

fn dense_taylor_negate(value: DenseTaylorJet) -> DenseTaylorJet {
    let mut out = DenseTaylorJet::zero(value.dimension, value.order);
    for (index, component) in value.coefficients.iter().enumerate() {
        let negated = component
            .as_ref()
            .map(|component| (value.factors[index], symbolic_negate(component)));
        out.set(index, negated);
    }
    if out.coefficients[0].is_none() {
        out.coefficients[0] = Some("0.0".to_string());
    }
    out
}

fn dense_taylor_scale(value: DenseTaylorJet, scalar: &str) -> DenseTaylorJet {
    let mut out = DenseTaylorJet::zero(value.dimension, value.order);
    for (index, component) in value.coefficients.iter().enumerate() {
        let scaled = component
            .as_ref()
            .map(|component| (value.factors[index], symbolic_multiply(component, scalar)));
        out.set(index, scaled);
    }
    if out.coefficients[0].is_none() {
        out.coefficients[0] = Some("0.0".to_string());
    }
    out
}

fn dense_taylor_multiply(left: DenseTaylorJet, right: DenseTaylorJet) -> DenseTaylorJet {
    let dimension = left.dimension;
    let order = left.order;
    let slots = dense_taylor_slot_count(dimension);
    let mut terms: Vec<Vec<(Rational, String)>> = vec![Vec::new(); slots];
    for (left_index, left_component) in left.coefficients.iter().enumerate() {
        let Some(left_component) = left_component else {
            continue;
        };
        let left_counts = dense_taylor_counts(left_index, dimension);
        for (right_index, right_component) in right.coefficients.iter().enumerate() {
            let Some(right_component) = right_component else {
                continue;
            };
            let right_counts = dense_taylor_counts(right_index, dimension);
            let counts = left_counts
                .iter()
                .zip(right_counts)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>();
            if counts.iter().sum::<usize>() > order {
                continue;
            }
            terms[dense_taylor_index(&counts)].push((
                left.factors[left_index].times(right.factors[right_index]),
                symbolic_multiply(left_component, right_component),
            ));
        }
    }
    let mut out = DenseTaylorJet::zero(dimension, order);
    for (index, terms) in terms.iter().enumerate() {
        out.set(index, combine_dense_taylor_terms(terms));
    }
    if out.coefficients[0].is_none() {
        out.coefficients[0] = Some("0.0".to_string());
    }
    out
}

struct DenseTaylorCompositionPartitions<'a> {
    candidates: &'a [(usize, String, Rational, Vec<usize>)],
    order: usize,
    counts: &'a mut Vec<usize>,
    selected: &'a mut Vec<usize>,
    /// Per output index, the partition products of the derivative order
    /// being enumerated, each with the product of its candidates' factors
    /// over the multiplicity factorial.
    products: &'a mut Vec<Vec<(Rational, String)>>,
}

fn dense_taylor_composition_partitions(
    state: &mut DenseTaylorCompositionPartitions<'_>,
    start: usize,
    remaining: usize,
    product: Option<(Rational, String)>,
) {
    if remaining == 0 {
        let index = dense_taylor_index(state.counts);
        let mut multiplicity_factorial = 1usize;
        let mut run = 1usize;
        for pair in state.selected.windows(2) {
            if pair[0] == pair[1] {
                run += 1;
            } else {
                multiplicity_factorial *= (2..=run).product::<usize>();
                run = 1;
            }
        }
        multiplicity_factorial *= (2..=run).product::<usize>();
        let (factor, expression) =
            product.expect("composition partition has at least one factor");
        state.products[index].push((
            factor.over(Rational::new(multiplicity_factorial as i64, 1)),
            expression,
        ));
        return;
    }

    for candidate_index in start..state.candidates.len() {
        let (_, component, factor, candidate_counts) = &state.candidates[candidate_index];
        let component = component.clone();
        let factor = *factor;
        let candidate_counts = candidate_counts.clone();
        if state
            .counts
            .iter()
            .zip(&candidate_counts)
            .map(|(left, right)| left + right)
            .sum::<usize>()
            > state.order
        {
            continue;
        }
        for (count, added) in state.counts.iter_mut().zip(&candidate_counts) {
            *count += added;
        }
        state.selected.push(candidate_index);
        let next = match &product {
            Some((product_factor, product)) => (
                product_factor.times(factor),
                symbolic_multiply(product, &component),
            ),
            None => (factor, component),
        };
        dense_taylor_composition_partitions(state, candidate_index, remaining - 1, Some(next));
        state.selected.pop();
        for (count, added) in state.counts.iter_mut().zip(&candidate_counts) {
            *count -= added;
        }
    }
}

/// The composition `f ∘ g` in Faà di Bruno's form: each output coefficient
/// is `Σ_d stack[d] · B_d`, where `B_d` is the sum over the partitions of
/// order `d` of the products of `g`'s coefficients. `B_d` is pure input
/// work, formed before the leaf is called; after the call the coefficient
/// costs one multiply per order and a short sum. (Multiplying every
/// partition product by its stack entry and summing the terms of all orders
/// in one chain put the whole sum behind the call.)
fn dense_taylor_compose(input: DenseTaylorJet, stack: &str) -> DenseTaylorJet {
    let dimension = input.dimension;
    let order = input.order;
    let slots = dense_taylor_slot_count(dimension);
    let candidates = input
        .coefficients
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(index, component)| {
            component.as_ref().map(|component| {
                (
                    index,
                    component.clone(),
                    input.factors[index],
                    dense_taylor_counts(index, dimension),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut out = DenseTaylorJet::constant(format!("{stack}[0]"), dimension, order);
    let mut terms: Vec<Vec<(usize, (Rational, String))>> = vec![Vec::new(); slots];
    for derivative_order in 1..=order {
        let mut counts = vec![0usize; dimension];
        let mut selected = Vec::with_capacity(derivative_order);
        let mut products: Vec<Vec<(Rational, String)>> = vec![Vec::new(); slots];
        let mut state = DenseTaylorCompositionPartitions {
            candidates: &candidates,
            order,
            counts: &mut counts,
            selected: &mut selected,
            products: &mut products,
        };
        dense_taylor_composition_partitions(&mut state, 0, derivative_order, None);
        for (index, products) in products.iter().enumerate() {
            if let Some(sum) = combine_dense_taylor_terms(products) {
                terms[index].push((derivative_order, sum));
            }
        }
    }
    // The orders' sums are combined under one factor before the stack
    // entries multiply them, so a ratio between orders is applied to `B_d`
    // (input work) and never to the product with the leaf's result.
    for (index, terms) in terms.iter().enumerate().skip(1) {
        let Some(common) = most_common_factor(terms.iter().map(|(_, (factor, _))| *factor))
        else {
            continue;
        };
        let mut sum: Option<String> = None;
        for (derivative_order, (factor, expression)) in terms {
            let scaled = scaled_by(expression, factor.over(common));
            let term = symbolic_multiply(&format!("{stack}[{derivative_order}]"), &scaled);
            sum = symbolic_add_component(&sum, &Some(term));
        }
        out.set(index, sum.map(|sum| (common, sum)));
    }
    out
}

fn push_dense_taylor_declaration(
    source: &mut String,
    indentation: &str,
    name: &str,
    mutable: &str,
    value: &DenseTaylorJet,
    support: &[bool],
) {
    for (index, present) in support.iter().copied().enumerate() {
        if present {
            source.push_str(&format!(
                "{indentation}let {mutable}{name}_c{index}: f64 = {};\n",
                symbolic_component(&value.coefficients[index]),
            ));
        }
    }
}

fn push_dense_taylor_assignment(
    source: &mut String,
    indentation: &str,
    name: &str,
    value: &DenseTaylorJet,
    support: &[bool],
) {
    for (index, present) in support.iter().copied().enumerate() {
        if present {
            source.push_str(&format!(
                "{indentation}{name}_c{index} = {};\n",
                symbolic_component(&value.coefficients[index]),
            ));
        }
    }
}

fn materialize_dense_taylor(
    value: DenseTaylorJet,
    owner: &str,
    temporary_index: &mut usize,
    preludes: &mut Vec<String>,
) -> DenseTaylorJet {
    let name = format!("__row_program_{owner}_dense_tmp{}", *temporary_index);
    *temporary_index += 1;
    let support = value.support();
    let mut source = String::new();
    push_dense_taylor_declaration(&mut source, "", &name, "", &value, &support);
    preludes.push(source);
    DenseTaylorJet::reference(&name, &support, &value.factors, value.dimension, value.order)
}

struct DenseTaylorExpressionEnvironment<'a> {
    leaves: &'a [Leaf],
    constants: &'a HashSet<String>,
    bindings: &'a HashMap<String, DenseTaylorJet>,
    dimension: usize,
    order: usize,
}

fn dense_taylor_expression(
    expression: &ProgramExpr,
    owner: &str,
    environment: &DenseTaylorExpressionEnvironment<'_>,
    temporary_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<DenseTaylorJet> {
    let leaves = environment.leaves;
    let constants = environment.constants;
    let bindings = environment.bindings;
    let dimension = environment.dimension;
    let order = environment.order;
    let mut child = |expression: &ProgramExpr| {
        dense_taylor_expression(expression, owner, environment, temporary_index, preludes)
    };
    let value = match expression {
        ProgramExpr::Path(ident) => {
            return bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(ident, "dense row_program binding is not defined")
            });
        }
        ProgramExpr::Zero => DenseTaylorJet::zero(dimension, order),
        ProgramExpr::Neg(value) => dense_taylor_negate(child(value)?),
        ProgramExpr::Scale(value, scalar) => dense_taylor_scale(
            child(value)?,
            &symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?,
        ),
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            let point = value.value(0).unwrap_or_else(|| "0.0".to_string());
            value.coefficients[0] = Some(symbolic_add(
                &point,
                &symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?,
            ));
            value.factors[0] = Rational::ONE;
            value
        }
        ProgramExpr::Add(left, right) => dense_taylor_add(child(left)?, child(right)?),
        ProgramExpr::Mul(left, right) => dense_taylor_multiply(child(left)?, child(right)?),
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "dense compose input is not defined")
            })?;
            let suffix = *temporary_index;
            *temporary_index += 1;
            let stack = format!("__row_program_{owner}_dense_stack{suffix}");
            let point = input.value(0).unwrap_or_else(|| "0.0".to_string());
            let mut leaf_arguments = Vec::new();
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, SymbolicTarget::Rust)?);
            }
            let application = leaves[*leaf].rust_application_source(&point, &leaf_arguments);
            preludes.push(format!("let {stack} = {application};"));
            // The dense path composes on the point as written, scaled or
            // not. The order-2 emitter absorbs a scaled point into the outer
            // stack (a measured win on every host); the same rule here —
            // composing on the unscaled binding through a stack rescaled by
            // `s^d` — emitted the same arithmetic minus the sign multiplies
            // and lost 10–35% on three hosts: LLVM scheduled the
            // observed-scale inverse-power chain and eleven spills of its
            // results ahead of the probit call instead of overlapping them
            // with it (30% of the generated arm's samples on the divide and
            // its consumers), a schedule the scaled point's own coefficients
            // happen not to provoke (#932).
            dense_taylor_compose(input, &stack)
        }
    };
    Ok(materialize_dense_taylor(
        value,
        owner,
        temporary_index,
        preludes,
    ))
}

fn directional_expression(
    expression: &ProgramExpr,
    owner: &str,
    environment: &DirectionalExpressionEnvironment<'_>,
    bindings: &HashMap<String, DirectionalJet>,
    stack_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<DirectionalJet> {
    let leaves = environment.leaves;
    let constants = environment.constants;
    let dimension = environment.dimension;
    let fourth = environment.fourth;
    let mut child = |expression: &ProgramExpr| {
        directional_expression(
            expression,
            owner,
            environment,
            bindings,
            stack_index,
            preludes,
        )
    };
    match expression {
        ProgramExpr::Path(ident) => bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
            syn::Error::new_spanned(ident, "directional row_program binding is not defined")
        }),
        ProgramExpr::Zero => Ok(DirectionalJet::zero(dimension)),
        ProgramExpr::Neg(value) => {
            let value = directional_negate(child(value)?);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = child(value)?;
            let scalar = symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?;
            let value = directional_scale(value, &scalar);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            value.base.value = symbolic_add(
                &value.base.value,
                &symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?,
            );
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Add(left, right) => {
            let left = child(left)?;
            let right = child(right)?;
            let value = directional_add(left, right);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Mul(left, right) => {
            let left = child(left)?;
            let right = child(right)?;
            let value = directional_multiply(left, right, fourth);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "directional compose input is not defined")
            })?;
            let suffix = *stack_index;
            *stack_index += 1;
            let stack = format!("{owner}_directional_stack{suffix}");
            let mut leaf_arguments = Vec::new();
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, SymbolicTarget::Rust)?);
            }
            let application =
                leaves[*leaf].rust_application_source(&input.base.value, &leaf_arguments);
            preludes.push(format!("let {stack} = {application};"));

            let base = symbolic_compose_jet(input.base.clone(), &stack, 0);
            let first = symbolic_compose_jet(input.base.clone(), &stack, 1);
            let u = symbolic_multiply_jets(first.clone(), input.u.clone());
            if !fourth {
                let value = DirectionalJet {
                    base,
                    u,
                    v: SymbolicJet::zero(dimension),
                    uv: SymbolicJet::zero(dimension),
                };
                return Ok(materialize_directional(
                    value,
                    owner,
                    fourth,
                    stack_index,
                    preludes,
                ));
            }
            let v = symbolic_multiply_jets(first.clone(), input.v.clone());
            let second = symbolic_compose_jet(input.base, &stack, 2);
            let uv = symbolic_add_jets(
                symbolic_multiply_jets(symbolic_multiply_jets(second, input.u), input.v),
                symbolic_multiply_jets(first, input.uv),
            );
            Ok(materialize_directional(
                DirectionalJet { base, u, v, uv },
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
    }
}

struct SymbolicLocal {
    name: String,
    mutable: bool,
    value: SymbolicJet,
    preludes: Vec<String>,
}

struct SymbolicAssignment {
    target: String,
    value: SymbolicJet,
    preludes: Vec<String>,
}

enum SymbolicStatement {
    Local(SymbolicLocal),
    If {
        condition: String,
        assignments: Vec<SymbolicAssignment>,
    },
}

struct SymbolicSchedule {
    statements: Vec<SymbolicStatement>,
    result: SymbolicJet,
    result_preludes: Vec<String>,
    mutable_support: HashMap<String, SymbolicSupport>,
    assigned: HashSet<String>,
    witness_values: Vec<String>,
}

struct DenseTaylorLocal {
    name: String,
    mutable: bool,
    value: DenseTaylorJet,
    preludes: Vec<String>,
}

struct DenseTaylorAssignment {
    target: String,
    value: DenseTaylorJet,
    preludes: Vec<String>,
}

enum DenseTaylorStatement {
    Local(DenseTaylorLocal),
    If {
        condition: String,
        assignments: Vec<DenseTaylorAssignment>,
    },
}

struct DenseTaylorSchedule {
    statements: Vec<DenseTaylorStatement>,
    result: DenseTaylorJet,
    root_compose_stack: Option<String>,
    result_preludes: Vec<String>,
    mutable_support: HashMap<String, Vec<bool>>,
    assigned: HashSet<String>,
}

fn include_dense_taylor_support(support: &mut [bool], value: &DenseTaylorJet) {
    for (present, component) in support.iter_mut().zip(&value.coefficients) {
        *present |= component.is_some();
    }
}

fn dense_taylor_schedule(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    order: usize,
    specialize_root_compose: bool,
) -> Result<DenseTaylorSchedule> {
    let dimension = primaries.len();
    let mut bindings = HashMap::<String, DenseTaylorJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            DenseTaylorJet::primary(&primary.to_string(), axis, dimension, order),
        );
    }
    let mut mutable_support = HashMap::<String, Vec<bool>>::new();
    let mut assigned = HashSet::new();
    let mut dense_statements = Vec::new();
    let mut temporary_index = 0;
    for statement in statements {
        match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let mut preludes = Vec::new();
                let environment = DenseTaylorExpressionEnvironment {
                    leaves,
                    constants,
                    bindings: &bindings,
                    dimension,
                    order,
                };
                let mut value = dense_taylor_expression(
                    value,
                    &name.to_string(),
                    &environment,
                    &mut temporary_index,
                    &mut preludes,
                )?;
                if *mutable {
                    value = value.normalized();
                }
                let support = value.support();
                if *mutable {
                    mutable_support.insert(name.to_string(), support.clone());
                }
                bindings.insert(
                    name.to_string(),
                    DenseTaylorJet::reference(
                        &name.to_string(),
                        &support,
                        &value.factors,
                        dimension,
                        order,
                    ),
                );
                dense_statements.push(DenseTaylorStatement::Local(DenseTaylorLocal {
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
                let mut dense_assignments = Vec::new();
                for (target_name, value) in assignments {
                    assigned.insert(target_name.to_string());
                    let mut preludes = Vec::new();
                    let environment = DenseTaylorExpressionEnvironment {
                        leaves,
                        constants,
                        bindings: &bindings,
                        dimension,
                        order,
                    };
                    let value = dense_taylor_expression(
                        value,
                        &target_name.to_string(),
                        &environment,
                        &mut temporary_index,
                        &mut preludes,
                    )?
                    .normalized();
                    let support = mutable_support
                        .get_mut(&target_name.to_string())
                        .expect("validated mutable dense Taylor target");
                    include_dense_taylor_support(support, &value);
                    bindings.insert(
                        target_name.to_string(),
                        DenseTaylorJet::reference(
                            &target_name.to_string(),
                            support,
                            &value.factors,
                            dimension,
                            order,
                        ),
                    );
                    dense_assignments.push(DenseTaylorAssignment {
                        target: target_name.to_string(),
                        value,
                        preludes,
                    });
                }
                dense_statements.push(DenseTaylorStatement::If {
                    condition: symbolic_scalar(condition, constants, SymbolicTarget::Rust)?,
                    assignments: dense_assignments,
                });
            }
        }
    }
    let mut result_preludes = Vec::new();
    let (result, root_compose_stack) = match result {
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } if specialize_root_compose && order == 3 => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "dense root compose input is not defined")
            })?;
            let stack = "__row_program_result_dense_root_stack".to_string();
            let point = input.value(0).unwrap_or_else(|| "0.0".to_string());
            let mut leaf_arguments = Vec::new();
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, SymbolicTarget::Rust)?);
            }
            let application = leaves[*leaf].rust_application_source(&point, &leaf_arguments);
            result_preludes.push(format!("let {stack} = {application};"));
            (input, Some(stack))
        }
        _ => {
            let environment = DenseTaylorExpressionEnvironment {
                leaves,
                constants,
                bindings: &bindings,
                dimension,
                order,
            };
            (
                dense_taylor_expression(
                    result,
                    "result",
                    &environment,
                    &mut temporary_index,
                    &mut result_preludes,
                )?,
                None,
            )
        }
    };
    Ok(DenseTaylorSchedule {
        statements: dense_statements,
        result,
        root_compose_stack,
        result_preludes,
        mutable_support,
        assigned,
    })
}

/// One emitted line or `if` block of a direct lowering's body.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ItemKind {
    /// A pure `let`/`double` definition: placed at the innermost gate that
    /// dominates every use of it and sunk to its first use there.
    Definition,
    /// A leaf call (`let stack = leaf(...)`, or CUDA's `double stack[3];`
    /// filled by `leaf(..., stack);`): placed like a definition, and issued
    /// as early as its inputs allow, so independent calls sit next to each
    /// other with nothing live between them.
    Call,
    /// An `if` block: an anchor, kept in program order.
    Gate,
    /// Anything else (the result lines, the CUDA closing brace): an anchor,
    /// kept in program order.
    Anchor,
}

struct EmittedItem {
    text: String,
    /// The name this item defines, for definitions and calls.
    defines: Option<String>,
    /// The mutable locals a gate block assigns (`name = ...;`).
    assigns: Vec<String>,
    references: Vec<String>,
    kind: ItemKind,
}

/// Where one definition or call is emitted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    /// At the top level, before the first anchor that needs it.
    Top,
    /// At the top of the one gate block that consumes it.
    Inside(usize),
    /// Nothing reaches an anchor through it: not emitted at all.
    Dead,
}

fn identifier_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| !(character.is_alphanumeric() || character == '_'))
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

/// The callee of a call expression `callee(...)` starting at `rhs`, when the
/// expression is one; a bare path with or without a module qualifier.
fn call_callee(rhs: &str) -> Option<&str> {
    let rhs = rhs.trim_start();
    let end = rhs
        .find(|character: char| !(character.is_alphanumeric() || character == '_' || character == ':'))
        .unwrap_or(rhs.len());
    let callee = &rhs[..end];
    let first = callee.chars().next()?;
    ((first.is_alphabetic() || first == '_') && rhs[end..].trim_start().starts_with('('))
        .then_some(callee)
}

/// Whether a definition's right-hand side is a call (`= leaf(...)`, with or
/// without a module path).
fn is_call_definition(line: &str) -> bool {
    let Some((_, rhs)) = line.split_once(" = ") else {
        return false;
    };
    call_callee(rhs).is_some()
}

/// A CUDA leaf-call statement `leaf(point, args..., stack);`.
fn is_cuda_call_statement(line: &str) -> bool {
    let trimmed = line.trim_start();
    !trimmed.starts_with("let ")
        && !trimmed.starts_with("double ")
        && !trimmed.starts_with("if ")
        && trimmed.ends_with(");")
        && !trimmed.contains(" = ")
        && call_callee(trimmed).is_some()
}

/// The stack a bare CUDA declaration `double stack[3];` declares, filled by
/// the leaf call on the next line.
fn cuda_stack_declaration(line: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix("double ")?;
    if rest.contains('=') || !rest.ends_with("];") {
        return None;
    }
    let name: String = rest
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The name a top-level `let`/`let mut`/`double` line binds. The result
/// lines (`__row_program_*`) are anchors, never definitions.
fn defined_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = if let Some(rest) = trimmed.strip_prefix("let mut ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("let ") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("double ") {
        rest
    } else {
        return None;
    };
    let name: String = rest
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty() && !name.starts_with("__row_program_")).then_some(name)
}

/// The mutable local a gate-block line assigns (`name = ...;`), if any.
fn assigned_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("let ") || trimmed.starts_with("double ") || trimmed.starts_with("if ") {
        return None;
    }
    let (lhs, _) = trimmed.split_once(" = ")?;
    let bare = lhs
        .chars()
        .all(|character| character.is_alphanumeric() || character == '_');
    (bare && !lhs.is_empty()).then(|| lhs.to_string())
}

/// Parse a body into items: one per top-level line, one per `if` block, one
/// per CUDA declaration-plus-call pair.
fn emitted_items(body: &str) -> Vec<EmittedItem> {
    let lines: Vec<&str> = body.lines().collect();
    let mut items = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        if line.trim_start().starts_with("if ") {
            let mut text = String::new();
            let mut assigns = Vec::new();
            loop {
                text.push_str(lines[index]);
                text.push('\n');
                if let Some(name) = assigned_name(lines[index]) {
                    assigns.push(name);
                }
                let closed = lines[index] == "    }";
                index += 1;
                if closed || index >= lines.len() {
                    break;
                }
            }
            let references = identifier_tokens(&text);
            items.push(EmittedItem {
                text,
                defines: None,
                assigns,
                references,
                kind: ItemKind::Gate,
            });
            continue;
        }
        if let Some(name) = cuda_stack_declaration(line)
            && index + 1 < lines.len()
            && is_cuda_call_statement(lines[index + 1])
            && identifier_tokens(lines[index + 1]).contains(&name)
        {
            let text = format!("{line}\n{}\n", lines[index + 1]);
            let references = identifier_tokens(&text);
            items.push(EmittedItem {
                text,
                defines: Some(name),
                assigns: Vec::new(),
                references,
                kind: ItemKind::Call,
            });
            index += 2;
            continue;
        }
        let text = format!("{line}\n");
        let references = identifier_tokens(&text);
        let (defines, kind) = match defined_name(line) {
            Some(name) if is_call_definition(line.trim_start()) => (Some(name), ItemKind::Call),
            Some(name) => (Some(name), ItemKind::Definition),
            None => (None, ItemKind::Anchor),
        };
        items.push(EmittedItem {
            text,
            defines,
            assigns: Vec::new(),
            references,
            kind,
        });
        index += 1;
    }
    items
}

/// Every anchor (gate or result line) that reads item `index`, directly or
/// through other definitions and calls. Definitions only reference earlier
/// items, so the walk over users terminates.
fn consuming_anchors(
    index: usize,
    items: &[EmittedItem],
    users: &[Vec<usize>],
    memo: &mut Vec<Option<Vec<usize>>>,
) -> Vec<usize> {
    if let Some(anchors) = &memo[index] {
        return anchors.clone();
    }
    let mut anchors = Vec::new();
    for &user in &users[index] {
        match items[user].kind {
            ItemKind::Gate | ItemKind::Anchor => {
                if !anchors.contains(&user) {
                    anchors.push(user);
                }
            }
            ItemKind::Definition | ItemKind::Call => {
                for anchor in consuming_anchors(user, items, users, memo) {
                    if !anchors.contains(&anchor) {
                        anchors.push(anchor);
                    }
                }
            }
        }
    }
    anchors.sort_unstable();
    memo[index] = Some(anchors.clone());
    anchors
}

/// Re-emit a direct lowering's body with every definition and leaf call placed
/// where the row's control flow wants it.
///
/// The symbolic schedule writes each statement's value, gradient and Hessian
/// channels together, in statement order, before every gate. Three things are
/// wrong with that as an instruction schedule, and each was found in the
/// release disassembly of a row the hand kernel beat (#932):
///
/// * **Work exclusive to one gate ran outside it.** The location-scale row's
///   event term (its index jet, its curvature coefficients, its composition
///   point) is read only under the event gate, yet was computed on every row;
///   the entry term's `exp` likewise. Every definition and call is therefore
///   placed at the innermost gate that dominates all of its uses: what only
///   one gate reads is emitted at the top of that gate's block, and a row on
///   which the gate is closed never pays for it. Nothing outside can name it,
///   so the move is invisible except in the instruction stream.
/// * **Leaf calls were separated by pure arithmetic and by each other's
///   consumers.** A call to `exp` or a probit kernel spills every live value
///   around it (the vector registers are all caller-saved), and two calls
///   issued back to back overlap their latencies while two calls a gate apart
///   do not. The hand kernels issue every transcendental first, when nothing
///   is live. Within its placement, a call is therefore issued as early as
///   its inputs allow -- right after the last gate whose assignment it reads,
///   or at the top -- with only its own input chain emitted before it, so
///   independent calls are adjacent.
/// * **Work a call does not feed was scheduled behind the call.** A
///   definition sinks to just before the first anchor that mentions it, but
///   never past a run of adjacent calls it does not depend on: everything
///   computable before the run is issued before it, in program order, as
///   every hand kernel writes it. Sinking such work behind a call delays it
///   by the call's whole instruction stream and hands LLVM's SLP vectoriser
///   a block in which call-independent chains sit beside call results; on
///   the rigid Bernoulli row it paired the observed-scale derivative chain
///   (a divide and five multiplies) with the probit stack and serialised it
///   behind the call, 0.98 → 0.81 against the hand kernel on one host and
///   0.75 on another, with an instruction histogram identical to the fast
///   version's. What depends on a call of the run still sinks behind the
///   run, so independent calls stay adjacent.
///
/// A definition or call nothing reaches an anchor through is not emitted (a
/// supplied stack composes without inspecting its point, so the point's value
/// channel is dead); `primary_parameters` then names the primaries whose
/// value the body never reads. Gates and the result lines keep program order.
///
/// A definition that reads a mutable local reads the state at its place in
/// the program: it is never sunk past, placed inside a gate behind, or
/// flushed across a gate that reassigns that local.
fn schedule_direct_lowering(body: &str) -> String {
    let mut items = emitted_items(body);
    let count = items.len();

    // Who defines each name, and who reads each item.
    let mut by_name = HashMap::<String, usize>::new();
    for (index, item) in items.iter().enumerate() {
        if let Some(name) = &item.defines {
            by_name.insert(name.clone(), index);
        }
    }
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (index, item) in items.iter().enumerate() {
        for reference in &item.references {
            if let Some(&defining) = by_name.get(reference)
                && defining != index
                && !users[defining].contains(&index)
            {
                users[defining].push(index);
            }
        }
    }
    // The gates after item `index` (in program order) that reassign a
    // mutable the item reads: the item's value is the state before them, so
    // it must be emitted before the first of them.
    let reassigning_gates = |index: usize| -> Vec<usize> {
        items
            .iter()
            .enumerate()
            .filter(|(gate, item)| {
                *gate > index
                    && item.kind == ItemKind::Gate
                    && item
                        .assigns
                        .iter()
                        .any(|assigned| items[index].references.contains(assigned))
            })
            .map(|(gate, _)| gate)
            .collect()
    };
    let mut memo = vec![None; count];
    let mut placement = vec![Placement::Top; count];
    for index in 0..count {
        if !matches!(items[index].kind, ItemKind::Definition | ItemKind::Call) {
            continue;
        }
        let anchors = consuming_anchors(index, &items, &users, &mut memo);
        placement[index] = match anchors.as_slice() {
            [] => Placement::Dead,
            [gate]
                if items[*gate].kind == ItemKind::Gate
                    && index < *gate
                    && reassigning_gates(index).iter().all(|other| other >= gate) =>
            {
                Placement::Inside(*gate)
            }
            _ => Placement::Top,
        };
    }

    // The pending definitions an item reads, directly or through other pending
    // definitions, in program order (a definition reads only earlier ones).
    fn pending_inputs<'a>(
        references: &'a [String],
        items: &'a [EmittedItem],
        pending: &[usize],
        by_name: &HashMap<String, usize>,
    ) -> Vec<usize> {
        let mut found = Vec::new();
        let mut stack: Vec<&'a str> = references.iter().map(String::as_str).collect();
        while let Some(reference) = stack.pop() {
            if let Some(&defining) = by_name.get(reference)
                && pending.contains(&defining)
                && !found.contains(&defining)
            {
                found.push(defining);
                stack.extend(items[defining].references.iter().map(String::as_str));
            }
        }
        found.sort_unstable();
        found
    }

    let mut order: Vec<usize> = Vec::with_capacity(count);
    let mut pending: Vec<usize> = Vec::new();
    let mut deferred: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut last_call: Option<usize> = None;
    let mut call_inputs = HashSet::<usize>::new();
    let position = |order: &[usize], index: usize| order.iter().position(|&i| i == index);

    for index in 0..count {
        match (items[index].kind, placement[index]) {
            (ItemKind::Definition | ItemKind::Call, Placement::Dead) => {}
            (ItemKind::Definition | ItemKind::Call, Placement::Inside(gate)) => {
                deferred[gate].push(index);
            }
            (ItemKind::Definition, Placement::Top) => pending.push(index),
            (ItemKind::Call, Placement::Top) => {
                // Issued right after the last emitted item it reads -- an
                // earlier definition or call, or a gate assigning a mutable it
                // reads -- and its own pending input chain goes with it.
                let inputs = pending_inputs(&items[index].references, &items, &pending, &by_name);
                let mut names: Vec<&str> =
                    items[index].references.iter().map(String::as_str).collect();
                for &input in &inputs {
                    names.extend(items[input].references.iter().map(String::as_str));
                }
                // The calls this one reads, directly or through its input
                // chain: it cannot join their run, and what their results
                // feed is issued between them and it.
                let dependency_calls: HashSet<usize> = names
                    .iter()
                    .filter_map(|name| by_name.get(*name).copied())
                    .filter(|&defining| items[defining].kind == ItemKind::Call)
                    .collect();
                let mut barrier: Option<usize> = None;
                for name in names {
                    if let Some(&defining) = by_name.get(name)
                        && let Some(at) = position(&order, defining)
                    {
                        barrier = Some(barrier.map_or(at, |current| current.max(at)));
                    }
                    for (gate, item) in items.iter().enumerate() {
                        if item.kind == ItemKind::Gate
                            && item.assigns.iter().any(|assigned| assigned == name)
                            && let Some(at) = position(&order, gate)
                        {
                            barrier = Some(barrier.map_or(at, |current| current.max(at)));
                        }
                    }
                }
                // Calls stay in program order among themselves: this one
                // follows the last call issued, so independent calls form
                // one adjacent run rather than leapfrogging each other.
                let mut at = barrier.map_or(0, |barrier| barrier + 1);
                if let Some(last) = last_call
                    && let Some(after) = position(&order, last)
                {
                    at = at.max(after + 1);
                }
                // The run of adjacent independent calls this one joins: the
                // calls and their own input chains immediately before `at`,
                // stopping at a call this one depends on.
                let mut run_start = at;
                while run_start > 0 {
                    let previous = order[run_start - 1];
                    let joins = (items[previous].kind == ItemKind::Call
                        && !dependency_calls.contains(&previous))
                        || call_inputs.contains(&previous);
                    if !joins {
                        break;
                    }
                    run_start -= 1;
                }
                // Every pending definition computable before the run is
                // issued before it, in program order; what depends on a call
                // of the run keeps sinking to its first use behind the run.
                let mut flush: Vec<usize> = Vec::new();
                for &candidate in &pending {
                    if inputs.contains(&candidate) {
                        continue;
                    }
                    let item = &items[candidate];
                    let available = item.references.iter().all(|name| match by_name.get(name) {
                        Some(&defining) if defining != candidate => {
                            flush.contains(&defining)
                                || position(&order, defining).is_some_and(|p| p < run_start)
                        }
                        _ => true,
                    });
                    // A mutable it reads must be in the same state as at its
                    // place in the program: every gate assigning that name is
                    // on the same side of the run as in program order.
                    let gates_agree = items.iter().enumerate().all(|(gate, other)| {
                        other.kind != ItemKind::Gate
                            || !other.assigns.iter().any(|assigned| item.references.contains(assigned))
                            || if gate < candidate {
                                position(&order, gate).is_some_and(|p| p < run_start)
                            } else {
                                position(&order, gate).is_none_or(|p| p >= run_start)
                            }
                    });
                    if available && gates_agree {
                        flush.push(candidate);
                    }
                }
                pending.retain(|candidate| !inputs.contains(candidate) && !flush.contains(candidate));
                at += flush.len();
                order.splice(run_start..run_start, flush);
                call_inputs.extend(inputs.iter().copied());
                let mut block = inputs;
                block.push(index);
                order.splice(at..at, block);
                last_call = Some(index);
            }
            (ItemKind::Gate, _) => {
                // The gate's exclusive items open its block: calls first,
                // each behind its own inputs, then the rest in program order.
                let inside = std::mem::take(&mut deferred[index]);
                let mut inside_pending: Vec<usize> = inside
                    .iter()
                    .copied()
                    .filter(|&i| items[i].kind == ItemKind::Definition)
                    .collect();
                // What none of the block's calls feed is computable at its
                // top and is issued there, before the calls, exactly as at the
                // top level; the rest sinks behind the calls.
                let mut inside_order = Vec::new();
                for &candidate in &inside_pending {
                    let independent = items[candidate].references.iter().all(|name| {
                        match by_name.get(name) {
                            Some(&defining) if defining != candidate && inside.contains(&defining) => {
                                inside_order.contains(&defining)
                            }
                            _ => true,
                        }
                    });
                    if independent {
                        inside_order.push(candidate);
                    }
                }
                inside_pending.retain(|candidate| !inside_order.contains(candidate));
                for &i in &inside {
                    if items[i].kind != ItemKind::Call {
                        continue;
                    }
                    let inputs = pending_inputs(&items[i].references, &items, &inside_pending, &by_name);
                    inside_pending.retain(|candidate| !inputs.contains(candidate));
                    inside_order.extend(inputs);
                    inside_order.push(i);
                }
                inside_order.extend(inside_pending);
                let mut references: Vec<String> = items[index].references.clone();
                for &i in &inside {
                    references.extend(items[i].references.iter().cloned());
                }
                if !inside_order.is_empty() {
                    let text = std::mem::take(&mut items[index].text);
                    let (head, tail) = text.split_once('\n').expect("a gate block has an `if` line");
                    let mut rebuilt = format!("{head}\n");
                    for &i in &inside_order {
                        for line in items[i].text.lines() {
                            rebuilt.push_str("    ");
                            rebuilt.push_str(line);
                            rebuilt.push('\n');
                        }
                    }
                    rebuilt.push_str(tail);
                    items[index].text = rebuilt;
                }
                // What reads a mutable this gate reassigns, and precedes the
                // gate in the program, reads the state before it: issued here.
                let mut readers: Vec<String> = Vec::new();
                for &candidate in &pending {
                    if candidate < index
                        && items[index]
                            .assigns
                            .iter()
                            .any(|assigned| items[candidate].references.contains(assigned))
                        && let Some(name) = &items[candidate].defines
                    {
                        readers.push(name.clone());
                    }
                }
                references.extend(readers);
                let inputs = pending_inputs(&references, &items, &pending, &by_name);
                pending.retain(|candidate| !inputs.contains(candidate));
                order.extend(inputs);
                order.push(index);
            }
            (ItemKind::Anchor, _) => {
                let inputs = pending_inputs(&items[index].references, &items, &pending, &by_name);
                pending.retain(|candidate| !inputs.contains(candidate));
                order.extend(inputs);
                order.push(index);
            }
        }
    }
    // A definition nothing mentions keeps its place before the body's last
    // anchor (the result lines, or CUDA's closing brace), never after it.
    let last = order.pop();
    order.extend(pending);
    order.extend(last);
    order.iter().map(|index| items[*index].text.as_str()).collect()
}

fn push_preludes(source: &mut String, preludes: &[String], indentation: &str) {
    for prelude in preludes {
        for line in prelude.lines() {
            source.push_str(indentation);
            source.push_str(line);
            source.push('\n');
        }
    }
}

fn symbolic_component(component: &Option<String>) -> &str {
    component.as_deref().unwrap_or("0.0")
}

fn symbolic_schedule(
    primaries: &[Ident],
    constants: &HashSet<String>,
    signs: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    witnesses: &[Ident],
    target: SymbolicTarget,
) -> Result<SymbolicSchedule> {
    let dimension = primaries.len();
    let mut bindings = HashMap::<String, SymbolicJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            SymbolicJet::primary(&primary.to_string(), axis, dimension),
        );
    }
    let mut mutable_support = HashMap::<String, SymbolicSupport>::new();
    let mut assigned = HashSet::new();
    let mut symbolic_statements = Vec::new();
    let mutable_names = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Local {
                name, mutable: true, ..
            } => Some(name.to_string()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut aliases = HashMap::<String, ScaledAlias>::new();
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
                // An immutable scalar multiple or negation of another binding
                // that no gate reassigns is an alias a later compose absorbs.
                if !*mutable {
                    let alias = match value {
                        ProgramExpr::Scale(inner, scalar) => match inner.as_ref() {
                            ProgramExpr::Path(inner) => Some(ScaledAlias {
                                inner: inner.to_string(),
                                scalar: Some(scalar.clone()),
                            }),
                            _ => None,
                        },
                        ProgramExpr::Neg(inner) => match inner.as_ref() {
                            ProgramExpr::Path(inner) => Some(ScaledAlias {
                                inner: inner.to_string(),
                                scalar: None,
                            }),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(alias) = alias
                        && !mutable_names.contains(&alias.inner)
                    {
                        aliases.insert(name.to_string(), alias);
                    }
                }
                let value = symbolic_expression(
                    value,
                    &name.to_string(),
                    leaves,
                    constants,
                    signs,
                    &bindings,
                    &aliases,
                    target,
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
                    SymbolicJet::reference(&name.to_string(), &support, dimension),
                );
                symbolic_statements.push(SymbolicStatement::Local(SymbolicLocal {
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
                let mut symbolic_assignments = Vec::new();
                for (target_name, value) in assignments {
                    assigned.insert(target_name.to_string());
                    let mut preludes = Vec::new();
                    let value = symbolic_expression(
                        value,
                        &target_name.to_string(),
                        leaves,
                        constants,
                        signs,
                        &bindings,
                        &aliases,
                        target,
                        dimension,
                        &mut stack_index,
                        &mut preludes,
                    )?;
                    let support = mutable_support
                        .get_mut(&target_name.to_string())
                        .expect("validated mutable symbolic target");
                    support.include(&value);
                    bindings.insert(
                        target_name.to_string(),
                        SymbolicJet::reference(&target_name.to_string(), support, dimension),
                    );
                    symbolic_assignments.push(SymbolicAssignment {
                        target: target_name.to_string(),
                        value,
                        preludes,
                    });
                }
                symbolic_statements.push(SymbolicStatement::If {
                    condition: symbolic_scalar(condition, constants, target)?,
                    assignments: symbolic_assignments,
                });
            }
        }
    }
    let witness_values = witnesses
        .iter()
        .map(|witness| {
            bindings
                .get(&witness.to_string())
                .map(|jet| jet.value.clone())
                .ok_or_else(|| {
                    syn::Error::new_spanned(witness, "symbolic witness binding is not defined")
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut result_preludes = Vec::new();
    let result = symbolic_expression(
        result,
        "result",
        leaves,
        constants,
        signs,
        &bindings,
        &aliases,
        target,
        dimension,
        &mut stack_index,
        &mut result_preludes,
    )?;
    Ok(SymbolicSchedule {
        statements: symbolic_statements,
        result,
        result_preludes,
        mutable_support,
        assigned,
        witness_values,
    })
}

struct DirectionalLocal {
    name: String,
    mutable: bool,
    value: DirectionalJet,
    preludes: Vec<String>,
}

struct DirectionalAssignment {
    target: String,
    value: DirectionalJet,
    preludes: Vec<String>,
}

enum DirectionalStatement {
    Local(DirectionalLocal),
    If {
        condition: String,
        assignments: Vec<DirectionalAssignment>,
    },
}

struct DirectionalSchedule {
    statements: Vec<DirectionalStatement>,
    result: DirectionalJet,
    result_preludes: Vec<String>,
    mutable_support: HashMap<String, DirectionalSupport>,
    assigned: HashSet<String>,
}

fn directional_prefix(name: &str, component: &str) -> String {
    format!("__row_program_{name}_{component}")
}

fn directional_reference(
    name: &str,
    support: &DirectionalSupport,
    dimension: usize,
    fourth: bool,
) -> DirectionalJet {
    DirectionalJet {
        base: SymbolicJet::reference(&directional_prefix(name, "base"), &support.base, dimension),
        u: SymbolicJet::reference(&directional_prefix(name, "u"), &support.u, dimension),
        v: if fourth {
            SymbolicJet::reference(&directional_prefix(name, "vdir"), &support.v, dimension)
        } else {
            SymbolicJet::zero(dimension)
        },
        uv: if fourth {
            SymbolicJet::reference(&directional_prefix(name, "uv"), &support.uv, dimension)
        } else {
            SymbolicJet::zero(dimension)
        },
    }
}

fn directional_schedule(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    fourth: bool,
) -> Result<DirectionalSchedule> {
    let dimension = primaries.len();
    let expression_environment = DirectionalExpressionEnvironment {
        leaves,
        constants,
        dimension,
        fourth,
    };
    let mut bindings = HashMap::<String, DirectionalJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            DirectionalJet::primary(&primary.to_string(), axis, dimension, fourth),
        );
    }
    let mut mutable_support = HashMap::<String, DirectionalSupport>::new();
    let mut assigned = HashSet::new();
    let mut directional_statements = Vec::new();
    let mut stack_index = 0;
    for statement in statements {
        match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let mut preludes = Vec::new();
                let value = directional_expression(
                    value,
                    &name.to_string(),
                    &expression_environment,
                    &bindings,
                    &mut stack_index,
                    &mut preludes,
                )?;
                let support = value.support();
                if *mutable {
                    mutable_support.insert(name.to_string(), support.clone());
                }
                bindings.insert(
                    name.to_string(),
                    directional_reference(&name.to_string(), &support, dimension, fourth),
                );
                directional_statements.push(DirectionalStatement::Local(DirectionalLocal {
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
                let mut directional_assignments = Vec::new();
                for (target_name, value) in assignments {
                    assigned.insert(target_name.to_string());
                    let mut preludes = Vec::new();
                    let value = directional_expression(
                        value,
                        &target_name.to_string(),
                        &expression_environment,
                        &bindings,
                        &mut stack_index,
                        &mut preludes,
                    )?;
                    let support = mutable_support
                        .get_mut(&target_name.to_string())
                        .expect("validated mutable directional target");
                    support.include(&value);
                    bindings.insert(
                        target_name.to_string(),
                        directional_reference(&target_name.to_string(), support, dimension, fourth),
                    );
                    directional_assignments.push(DirectionalAssignment {
                        target: target_name.to_string(),
                        value,
                        preludes,
                    });
                }
                directional_statements.push(DirectionalStatement::If {
                    condition: symbolic_scalar(condition, constants, SymbolicTarget::Rust)?,
                    assignments: directional_assignments,
                });
            }
        }
    }
    let mut result_preludes = Vec::new();
    let result = directional_expression(
        result,
        "result",
        &expression_environment,
        &bindings,
        &mut stack_index,
        &mut result_preludes,
    )?;
    Ok(DirectionalSchedule {
        statements: directional_statements,
        result,
        result_preludes,
        mutable_support,
        assigned,
    })
}

fn push_symbolic_declaration(
    source: &mut String,
    indentation: &str,
    prefix: &str,
    mutable: &str,
    value: &SymbolicJet,
    support: &SymbolicSupport,
) {
    let dimension = value.gradient.len();
    source.push_str(&format!(
        "{indentation}let {mutable}{prefix}_v: f64 = {};\n",
        value.value
    ));
    for axis in 0..dimension {
        if support.gradient[axis] {
            source.push_str(&format!(
                "{indentation}let {mutable}{prefix}_g{axis}: f64 = {};\n",
                symbolic_component(&value.gradient[axis]),
            ));
        }
        for other in axis..dimension {
            let index = axis * dimension + other;
            if support.hessian[index] {
                source.push_str(&format!(
                    "{indentation}let {mutable}{prefix}_h{axis}_{other}: f64 = {};\n",
                    symbolic_component(&value.hessian[index]),
                ));
            }
        }
    }
}

fn push_symbolic_assignment(
    source: &mut String,
    indentation: &str,
    prefix: &str,
    value: &SymbolicJet,
    support: &SymbolicSupport,
) {
    let dimension = value.gradient.len();
    source.push_str(&format!("{indentation}{prefix}_v = {};\n", value.value));
    for axis in 0..dimension {
        if support.gradient[axis] {
            source.push_str(&format!(
                "{indentation}{prefix}_g{axis} = {};\n",
                symbolic_component(&value.gradient[axis]),
            ));
        }
        for other in axis..dimension {
            let index = axis * dimension + other;
            if support.hessian[index] {
                source.push_str(&format!(
                    "{indentation}{prefix}_h{axis}_{other} = {};\n",
                    symbolic_component(&value.hessian[index]),
                ));
            }
        }
    }
}

fn push_directional_declaration(
    source: &mut String,
    indentation: &str,
    name: &str,
    mutable: &str,
    value: &DirectionalJet,
    support: &DirectionalSupport,
    fourth: bool,
) {
    push_symbolic_declaration(
        source,
        indentation,
        &directional_prefix(name, "base"),
        mutable,
        &value.base,
        &support.base,
    );
    push_symbolic_declaration(
        source,
        indentation,
        &directional_prefix(name, "u"),
        mutable,
        &value.u,
        &support.u,
    );
    if fourth {
        push_symbolic_declaration(
            source,
            indentation,
            &directional_prefix(name, "vdir"),
            mutable,
            &value.v,
            &support.v,
        );
        push_symbolic_declaration(
            source,
            indentation,
            &directional_prefix(name, "uv"),
            mutable,
            &value.uv,
            &support.uv,
        );
    }
}

fn push_directional_assignment(
    source: &mut String,
    indentation: &str,
    name: &str,
    value: &DirectionalJet,
    support: &DirectionalSupport,
    fourth: bool,
) {
    push_symbolic_assignment(
        source,
        indentation,
        &directional_prefix(name, "base"),
        &value.base,
        &support.base,
    );
    push_symbolic_assignment(
        source,
        indentation,
        &directional_prefix(name, "u"),
        &value.u,
        &support.u,
    );
    if fourth {
        push_symbolic_assignment(
            source,
            indentation,
            &directional_prefix(name, "vdir"),
            &value.v,
            &support.v,
        );
        push_symbolic_assignment(
            source,
            indentation,
            &directional_prefix(name, "uv"),
            &value.uv,
            &support.uv,
        );
    }
}

fn dense_taylor_derivative(value: &DenseTaylorJet, axes: &[usize]) -> Option<String> {
    let mut counts = vec![0usize; value.dimension];
    for axis in axes {
        counts[*axis] += 1;
    }
    let index = dense_taylor_index(&counts);
    let component = value.coefficients[index].as_ref()?.clone();
    let factorial = counts
        .iter()
        .map(|count| Rational::factorial(*count))
        .fold(Rational::ONE, Rational::times);
    Some(scaled_by(&component, value.factors[index].times(factorial)))
}

fn dense_taylor_contracted_component(
    value: &DenseTaylorJet,
    axis: usize,
    other: usize,
    fourth: bool,
) -> String {
    let mut component = None;
    for direction_axis in 0..value.dimension {
        if fourth {
            for other_direction_axis in 0..value.dimension {
                let derivative = dense_taylor_derivative(
                    value,
                    &[axis, other, direction_axis, other_direction_axis],
                );
                let directed = derivative.map(|derivative| {
                    symbolic_multiply(
                        &symbolic_multiply(&derivative, &format!("direction_u[{direction_axis}]")),
                        &format!("direction_v[{other_direction_axis}]"),
                    )
                });
                component = symbolic_add_component(&component, &directed);
            }
        } else {
            let derivative = dense_taylor_derivative(value, &[axis, other, direction_axis]);
            let directed = derivative.map(|derivative| {
                symbolic_multiply(&derivative, &format!("direction_u[{direction_axis}]"))
            });
            component = symbolic_add_component(&component, &directed);
        }
    }
    symbolic_component(&component).to_string()
}

fn push_dense_taylor_derivative_array(
    source: &mut String,
    name: &str,
    value: &DenseTaylorJet,
    derivative_order: usize,
) {
    source.push_str(&format!("    let {name} = ["));
    for ones in 0..=derivative_order {
        if ones != 0 {
            source.push_str(", ");
        }
        let mut axes = vec![0usize; derivative_order - ones];
        axes.extend(std::iter::repeat_n(1usize, ones));
        source.push_str(symbolic_component(&dense_taylor_derivative(value, &axes)));
    }
    source.push_str("];\n");
}

fn push_dense_taylor_schedule_body(source: &mut String, schedule: &DenseTaylorSchedule) {
    for statement in &schedule.statements {
        match statement {
            DenseTaylorStatement::Local(local) => {
                push_preludes(source, &local.preludes, "    ");
                let mutable = if schedule.assigned.contains(&local.name) {
                    "mut "
                } else {
                    ""
                };
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable dense Taylor support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                push_dense_taylor_declaration(
                    source,
                    "    ",
                    &local.name,
                    mutable,
                    &local.value,
                    &support,
                );
            }
            DenseTaylorStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if {condition} {{\n"));
                for assignment in assignments {
                    push_preludes(source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable dense Taylor assignment support exists");
                    push_dense_taylor_assignment(
                        source,
                        "        ",
                        &assignment.target,
                        &assignment.value,
                        support,
                    );
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(source, &schedule.result_preludes, "    ");
}

fn rust_dense_taylor_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    fourth: bool,
) -> Result<syn::Block> {
    let dimension = primaries.len();
    let order = if fourth { 4 } else { 3 };
    let schedule = dense_taylor_schedule(
        primaries, constants, leaves, statements, result, order, true,
    )?;
    let mut source = "{\n".to_string();
    push_dense_taylor_schedule_body(&mut source, &schedule);
    if dimension == 2 {
        // The root-composition specialisation is the third-order surface's:
        // the inner jet's derivatives contracted in Faà di Bruno's form. The
        // fourth-order surface composes densely and contracts the four
        // degree-four coefficients, which is fewer multiplies than the
        // directional form at that order.
        if let Some(root_stack) = &schedule.root_compose_stack {
            assert!(!fourth, "the root composition is specialised at order 3 only");
            push_dense_taylor_derivative_array(&mut source, "inner_first", &schedule.result, 1);
            push_dense_taylor_derivative_array(&mut source, "inner_second", &schedule.result, 2);
            push_dense_taylor_derivative_array(&mut source, "inner_third", &schedule.result, 3);
            source.push_str(&format!(
                "    let inner_u = inner_first[0] * direction_u[0]\n\
                 \x20       + inner_first[1] * direction_u[1];\n\
                 \x20   std::array::from_fn(|axis| std::array::from_fn(|other| {{\n\
                 \x20       let offset = axis + other;\n\
                 \x20       let inner_a = inner_first[axis];\n\
                 \x20       let inner_b = inner_first[other];\n\
                 \x20       let inner_ab = inner_second[offset];\n\
                 \x20       let inner_au = inner_second[axis] * direction_u[0]\n\
                 \x20           + inner_second[axis + 1] * direction_u[1];\n\
                 \x20       let inner_bu = inner_second[other] * direction_u[0]\n\
                 \x20           + inner_second[other + 1] * direction_u[1];\n\
                 \x20       let inner_abu = inner_third[offset] * direction_u[0]\n\
                 \x20           + inner_third[offset + 1] * direction_u[1];\n\
                 \x20       {root_stack}[3] * inner_u * inner_a * inner_b\n\
                 \x20           + {root_stack}[2] * (inner_au * inner_b + inner_a * inner_bu\n\
                 \x20               + inner_u * inner_ab)\n\
                 \x20           + {root_stack}[1] * inner_abu\n\
                 \x20   }}))\n"
            ));
            source.push_str("}\n");
            return syn::parse_str(&source).map_err(|error| {
                syn::Error::new(
                    error.span(),
                    format!(
                        "failed to parse generated Rust dense root-compose third-order row program: {error}\n{source}"
                    ),
                )
            });
        }
    }
    if dimension == 2 {
        source.push_str("    let dense_derivatives = [");
        for ones in 0..=order {
            if ones != 0 {
                source.push_str(", ");
            }
            let mut axes = vec![0usize; order - ones];
            axes.extend(std::iter::repeat_n(1usize, ones));
            source.push_str(symbolic_component(&dense_taylor_derivative(
                &schedule.result,
                &axes,
            )));
        }
        source.push_str("];\n");
        if fourth {
            source.push_str(
                "    let direction_00 = direction_u[0] * direction_v[0];\n\
                 \x20   let direction_01 = direction_u[0] * direction_v[1]\n\
                 \x20       + direction_u[1] * direction_v[0];\n\
                 \x20   let direction_11 = direction_u[1] * direction_v[1];\n\
                 \x20   std::array::from_fn(|axis| std::array::from_fn(|other| {\n\
                 \x20       let offset = axis + other;\n\
                 \x20       dense_derivatives[offset] * direction_00\n\
                 \x20           + dense_derivatives[offset + 1] * direction_01\n\
                 \x20           + dense_derivatives[offset + 2] * direction_11\n\
                 \x20   }))\n",
            );
        } else {
            source.push_str(
                "    std::array::from_fn(|axis| std::array::from_fn(|other| {\n\
                 \x20       let offset = axis + other;\n\
                 \x20       dense_derivatives[offset] * direction_u[0]\n\
                 \x20           + dense_derivatives[offset + 1] * direction_u[1]\n\
                 \x20   }))\n",
            );
        }
    } else {
        source.push_str("    [\n");
        for axis in 0..dimension {
            source.push_str("        [");
            for other in 0..dimension {
                if other != 0 {
                    source.push_str(", ");
                }
                source.push_str(&dense_taylor_contracted_component(
                    &schedule.result,
                    axis,
                    other,
                    fourth,
                ));
            }
            source.push_str("],\n");
        }
        source.push_str("    ]\n");
    }
    source.push_str("}\n");
    let order = if fourth { "fourth" } else { "third" };
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!(
                "failed to parse generated Rust dense {order}-order row program: {error}\n{source}"
            ),
        )
    })
}

fn rust_dense_taylor_uncontracted_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    order: usize,
) -> Result<syn::Block> {
    if primaries.len() != 2 {
        return Err(syn::Error::new_spanned(
            &primaries[0],
            "uncontracted row_program surfaces currently require exactly two primaries",
        ));
    }
    let schedule = dense_taylor_schedule(
        primaries,
        constants,
        leaves,
        statements,
        result,
        order,
        order == 3,
    )?;
    let mut source = "{\n".to_string();
    push_dense_taylor_schedule_body(&mut source, &schedule);
    if order == 3
        && let Some(root_stack) = &schedule.root_compose_stack
    {
        push_dense_taylor_derivative_array(&mut source, "inner_first", &schedule.result, 1);
        push_dense_taylor_derivative_array(&mut source, "inner_second", &schedule.result, 2);
        push_dense_taylor_derivative_array(&mut source, "inner_third", &schedule.result, 3);
        source.push_str(&format!(
            "    std::array::from_fn(|axis_a| std::array::from_fn(|axis_b| {{\n\
             \x20       std::array::from_fn(|axis_c| {{\n\
             \x20           let inner_a = inner_first[axis_a];\n\
             \x20           let inner_b = inner_first[axis_b];\n\
             \x20           let inner_c = inner_first[axis_c];\n\
             \x20           let inner_ab = inner_second[axis_a + axis_b];\n\
             \x20           let inner_ac = inner_second[axis_a + axis_c];\n\
             \x20           let inner_bc = inner_second[axis_b + axis_c];\n\
             \x20           let inner_abc = inner_third[axis_a + axis_b + axis_c];\n\
             \x20           {root_stack}[3] * inner_a * inner_b * inner_c\n\
             \x20               + {root_stack}[2] * (inner_ab * inner_c\n\
             \x20                   + inner_ac * inner_b + inner_bc * inner_a)\n\
             \x20               + {root_stack}[1] * inner_abc\n\
             \x20       }})\n\
             \x20   }}))\n\
             }}\n"
        ));
        return syn::parse_str(&source).map_err(|error| {
            syn::Error::new(
                error.span(),
                format!(
                    "failed to parse generated Rust root-compose uncontracted order-3 row program: {error}\n{source}"
                ),
            )
        });
    }
    push_dense_taylor_derivative_array(&mut source, "derivative", &schedule.result, order);
    if order == 3 {
        source.push_str(
            "    [\n\
             \x20       [[derivative[0], derivative[1]], [derivative[1], derivative[2]]],\n\
             \x20       [[derivative[1], derivative[2]], [derivative[2], derivative[3]]],\n\
             \x20   ]\n\
             }\n",
        );
    } else {
        source.push_str(
            "    [\n\
             \x20       [\n\
             \x20           [[derivative[0], derivative[1]], [derivative[1], derivative[2]]],\n\
             \x20           [[derivative[1], derivative[2]], [derivative[2], derivative[3]]],\n\
             \x20       ],\n\
             \x20       [\n\
             \x20           [[derivative[1], derivative[2]], [derivative[2], derivative[3]]],\n\
             \x20           [[derivative[2], derivative[3]], [derivative[3], derivative[4]]],\n\
             \x20       ],\n\
             \x20   ]\n\
             }\n",
        );
    }
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!(
                "failed to parse generated Rust uncontracted order-{order} row program: {error}\n{source}"
            ),
        )
    })
}

fn rust_directional_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    fourth: bool,
) -> Result<syn::Block> {
    let dimension = primaries.len();
    let schedule = directional_schedule(primaries, constants, leaves, statements, result, fourth)?;
    let mut source = "{\n".to_string();
    for statement in &schedule.statements {
        match statement {
            DirectionalStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let mutable = if schedule.assigned.contains(&local.name) {
                    "mut "
                } else {
                    ""
                };
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable directional support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                push_directional_declaration(
                    &mut source,
                    "    ",
                    &local.name,
                    mutable,
                    &local.value,
                    &support,
                    fourth,
                );
            }
            DirectionalStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if {condition} {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable directional assignment support exists");
                    push_directional_assignment(
                        &mut source,
                        "        ",
                        &assignment.target,
                        &assignment.value,
                        support,
                        fourth,
                    );
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    let contracted = if fourth {
        &schedule.result.uv
    } else {
        &schedule.result.u
    };
    source.push_str("    [\n");
    for axis in 0..dimension {
        source.push_str("        [");
        for other in 0..dimension {
            if other != 0 {
                source.push_str(", ");
            }
            let (row, column) = if axis <= other {
                (axis, other)
            } else {
                (other, axis)
            };
            let index = row * dimension + column;
            source.push_str(symbolic_component(&contracted.hessian[index]));
        }
        source.push_str("],\n");
    }
    source.push_str("    ]\n}\n");
    let order = if fourth { "fourth" } else { "third" };
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!(
                "failed to parse generated Rust {order}-order contracted row program: {error}\n{source}"
            ),
        )
    })
}

fn rust_order2_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    signs: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    witnesses: &[Ident],
) -> Result<syn::Block> {
    let dimension = primaries.len();
    let schedule = symbolic_schedule(
        primaries,
        constants,
        signs,
        leaves,
        statements,
        result,
        witnesses,
        SymbolicTarget::Rust,
    )?;
    let mut source = "{\n".to_string();
    // Channels whose value is `0.0` on EVERY path reaching the statement being
    // emitted. A mutable channel starts here when its declaration is `0.0`, and
    // stays here only while every gate that assigns it assigns `0.0` too, so a
    // later `channel = 0.0` in a gate is writing the value the channel already
    // has on both edges of that branch. Together with the self-assignment case
    // below this is what keeps a gate from restating the channels its term does
    // not contribute to: an added term touches a handful of channels, and the
    // union support made it restate all of them (#932).
    let mut known_zero: HashSet<String> = HashSet::new();
    for statement in &schedule.statements {
        match statement {
            SymbolicStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let mutable = if schedule.assigned.contains(&local.name) {
                    "mut "
                } else {
                    ""
                };
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable symbolic support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                let mut declare = |source: &mut String, channel: String, value: &str| {
                    if value == "0.0" {
                        known_zero.insert(channel.clone());
                    }
                    source.push_str(&format!("    let {mutable}{channel}: f64 = {value};\n"));
                };
                declare(
                    &mut source,
                    format!("{}_v", local.name),
                    &local.value.value,
                );
                for axis in 0..dimension {
                    if support.gradient[axis] {
                        declare(
                            &mut source,
                            format!("{}_g{axis}", local.name),
                            symbolic_component(&local.value.gradient[axis]),
                        );
                    }
                    for other in axis..dimension {
                        let index = axis * dimension + other;
                        if support.hessian[index] {
                            declare(
                                &mut source,
                                format!("{}_h{axis}_{other}", local.name),
                                symbolic_component(&local.value.hessian[index]),
                            );
                        }
                    }
                }
            }
            SymbolicStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if {condition} {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable symbolic assignment support exists");
                    // An assignment a gate does not need to make. `channel =
                    // channel` restates the value the channel already holds on
                    // both edges, and `channel = 0.0` where the channel is `0.0`
                    // on every path into the gate does the same. Both come from
                    // the union support: a term contributes to a handful of
                    // channels and the union makes it name all of them. Skipping
                    // them leaves the channel's value untouched, which IS the
                    // value the skipped statement would have written, so the
                    // emitted schedule is unchanged in what it computes and is
                    // two thirds shorter on a three-term row program (#932).
                    let mut assign = |source: &mut String, channel: String, value: &str| {
                        if value == channel {
                            return;
                        }
                        if value == "0.0" {
                            if known_zero.contains(&channel) {
                                return;
                            }
                            // Zero on the taken edge only: the channel is no
                            // longer known-zero after the branch.
                            known_zero.remove(&channel);
                        } else {
                            known_zero.remove(&channel);
                        }
                        source.push_str(&format!("        {channel} = {value};\n"));
                    };
                    assign(
                        &mut source,
                        format!("{}_v", assignment.target),
                        &assignment.value.value,
                    );
                    for axis in 0..dimension {
                        if support.gradient[axis] {
                            assign(
                                &mut source,
                                format!("{}_g{axis}", assignment.target),
                                symbolic_component(&assignment.value.gradient[axis]),
                            );
                        }
                        for other in axis..dimension {
                            let index = axis * dimension + other;
                            if support.hessian[index] {
                                assign(
                                    &mut source,
                                    format!("{}_h{axis}_{other}", assignment.target),
                                    symbolic_component(&assignment.value.hessian[index]),
                                );
                            }
                        }
                    }
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    source.push_str(&format!(
        "    let __row_program_value: f64 = {};\n",
        schedule.result.value
    ));
    for axis in 0..dimension {
        source.push_str(&format!(
            "    let __row_program_g{axis}: f64 = {};\n",
            symbolic_component(&schedule.result.gradient[axis]),
        ));
        for other in axis..dimension {
            let index = axis * dimension + other;
            source.push_str(&format!(
                "    let __row_program_h{axis}_{other}: f64 = {};\n",
                symbolic_component(&schedule.result.hessian[index]),
            ));
        }
    }
    source.push_str("    (\n        __row_program_value,\n        [");
    for axis in 0..dimension {
        if axis != 0 {
            source.push_str(", ");
        }
        source.push_str(&format!("__row_program_g{axis}"));
    }
    source.push_str("],\n        [\n");
    for axis in 0..dimension {
        source.push_str("            [");
        for other in 0..dimension {
            if other != 0 {
                source.push_str(", ");
            }
            let (row, column) = if axis <= other {
                (axis, other)
            } else {
                (other, axis)
            };
            source.push_str(&format!("__row_program_h{row}_{column}"));
        }
        source.push_str("],\n");
    }
    source.push_str("        ],\n        [");
    for (index, witness) in schedule.witness_values.iter().enumerate() {
        if index != 0 {
            source.push_str(", ");
        }
        source.push_str(witness);
    }
    source.push_str("]\n    )\n}\n");
    // The result tuple is scheduled with the body: its lines are the anchors
    // that keep every channel and witness value live.
    let (open, body) = source
        .split_once('\n')
        .expect("the order-2 body opens with a brace");
    let source = format!("{open}\n{}", schedule_direct_lowering(body));
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!("failed to parse generated Rust order-2 row program: {error}\n{source}"),
        )
    })
}

fn cuda_source(
    name: &Ident,
    primaries: &[Ident],
    constants: &HashSet<String>,
    signs: &HashSet<String>,
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
    let schedule = symbolic_schedule(
        primaries,
        constants,
        signs,
        leaves,
        statements,
        result,
        &[],
        SymbolicTarget::Cuda,
    )?;

    let mut source = format!("__device__ __forceinline__ void {name}(\n        {parameters}) {{\n");
    for statement in &schedule.statements {
        match statement {
            SymbolicStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable symbolic support exists")
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
                            symbolic_component(&local.value.gradient[axis]),
                        ));
                    }
                    for other in axis..dimension {
                        let index = axis * dimension + other;
                        if support.hessian[index] {
                            source.push_str(&format!(
                                "    double {}_h{axis}_{other} = {};\n",
                                local.name,
                                symbolic_component(&local.value.hessian[index]),
                            ));
                        }
                    }
                }
            }
            SymbolicStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if ({condition}) {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable symbolic assignment support exists");
                    source.push_str(&format!(
                        "        {}_v = {};\n",
                        assignment.target, assignment.value.value,
                    ));
                    for axis in 0..dimension {
                        if support.gradient[axis] {
                            source.push_str(&format!(
                                "        {}_g{axis} = {};\n",
                                assignment.target,
                                symbolic_component(&assignment.value.gradient[axis]),
                            ));
                        }
                        for other in axis..dimension {
                            let index = axis * dimension + other;
                            if support.hessian[index] {
                                source.push_str(&format!(
                                    "        {}_h{axis}_{other} = {};\n",
                                    assignment.target,
                                    symbolic_component(&assignment.value.hessian[index]),
                                ));
                            }
                        }
                    }
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    source.push_str(&format!("    *row_value = {};\n", schedule.result.value));
    for axis in 0..dimension {
        source.push_str(&format!(
            "    row_gradient[{axis}] = {};\n",
            symbolic_component(&schedule.result.gradient[axis]),
        ));
        for other in axis..dimension {
            let index = axis * dimension + other;
            let component = symbolic_component(&schedule.result.hessian[index]);
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
    let (header, body) = source
        .split_once(") {\n")
        .expect("the CUDA signature closes before the body");
    let scheduled = schedule_direct_lowering(body);
    Ok(format!("{header}) {{\n{scheduled}"))
}

pub(crate) fn expand(input: Input) -> Result<TokenStream2> {
    let Input {
        visibility,
        name,
        primaries,
        constants,
        signs,
        emissions,
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
    let sign_names = signs
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
    for statement in &statements {
        match statement {
            Statement::Local { value, .. } => validate_supplied_stacks(value, &leaves)?,
            Statement::If { assignments, .. } => {
                for assignment in assignments {
                    validate_supplied_stacks(&assignment.1, &leaves)?;
                }
            }
        }
    }
    validate_supplied_stacks(&result, &leaves)?;
    for witness in &witnesses {
        if !bindings.contains(&witness.to_string()) {
            return Err(syn::Error::new_spanned(
                witness,
                "row_program witness is not a defined jet",
            ));
        }
    }
    let witness_count = witnesses.len();
    if emissions.witnesses && witnesses.is_empty() {
        return Err(syn::Error::new_spanned(
            &name,
            "row_program cannot emit a `witnesses` surface with no declared witnesses",
        ));
    }
    let dimension = primaries.len();

    let generic_function = if emissions.generic {
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
        quote! {
            #[inline(always)]
            #visibility fn #name<
                const __ROW_PROGRAM_DERIVATIVE_DIMENSION: usize,
                S: ::gam_math::jet_scalar::JetScalar<__ROW_PROGRAM_DERIVATIVE_DIMENSION>,
            >(
                #(#primaries: &S,)*
                #(#constants: f64),*
            ) -> (S, [f64; #witness_count]) {
                #(#rust_statements)*
                let emitted_row_program_value = #rust_result;
                (emitted_row_program_value, [#(#witness_values),*])
            }
        }
    } else {
        quote!()
    };

    let runtime_function = if emissions.runtime {
        let runtime_name = format_ident!("{}_runtime", name);
        let runtime_primary_bindings = primaries
            .iter()
            .map(|primary| quote!(let #primary = (*#primary).clone();));
        let runtime_statements = statements.iter().map(|statement| match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let value = rust_runtime_expression(value, &leaves);
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
                    let value = rust_runtime_expression(value, &leaves);
                    quote!(#target = #value;)
                });
                quote!(if #condition { #(#assignments)* })
            }
        });
        let runtime_result = rust_runtime_expression(&result, &leaves);
        let runtime_witness_values = witnesses.iter().map(|witness| quote!(#witness.value()));
        quote! {
            #[inline(always)]
            #visibility fn #runtime_name<'arena, S: ::gam_math::jet_scalar::RuntimeJetScalar<'arena>>(
                #(#primaries: &S,)*
                #(#constants: f64,)*
                __row_program_dimension: usize,
                __row_program_workspace: &'arena S::Workspace,
            ) -> (S, [f64; #witness_count]) {
                #(#runtime_primary_bindings)*
                #(#runtime_statements)*
                let emitted_row_program_value = #runtime_result;
                (emitted_row_program_value, [#(#runtime_witness_values),*])
            }
        }
    } else {
        quote!()
    };

    let order2_function = if emissions.order2 {
        let order2_name = format_ident!("{}_order2", name);
        let order2_body = rust_order2_body(
            &primaries,
            &constant_names,
            &sign_names,
            &leaves,
            &statements,
            &result,
            &witnesses,
        )?;
        let order2_primaries = primary_parameters(&primaries, &quote!(#order2_body));
        quote! {
            #[inline(always)]
            #visibility fn #order2_name(
                #(#order2_primaries: f64,)*
                #(#constants: f64),*
            ) -> (
                f64,
                [f64; #dimension],
                [[f64; #dimension]; #dimension],
                [f64; #witness_count],
            ) #order2_body
        }
    } else {
        quote!()
    };

    let third_function = if emissions.third {
        let third_name = format_ident!("{}_third_contracted", name);
        let third_body = if dimension <= 2 {
            rust_dense_taylor_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                false,
            )?
        } else {
            rust_directional_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                false,
            )?
        };
        let third_primaries = primary_parameters(&primaries, &quote!(#third_body));
        quote! {
            #[inline(always)]
            #visibility fn #third_name(
                #(#third_primaries: f64,)*
                #(#constants: f64,)*
                direction_u: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] #third_body
        }
    } else {
        quote!()
    };

    let fourth_function = if emissions.fourth {
        let fourth_name = format_ident!("{}_fourth_contracted", name);
        let fourth_body = if dimension <= 2 {
            rust_dense_taylor_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                true,
            )?
        } else {
            rust_directional_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                true,
            )?
        };
        let fourth_primaries = primary_parameters(&primaries, &quote!(#fourth_body));
        quote! {
            #[inline(always)]
            #visibility fn #fourth_name(
                #(#fourth_primaries: f64,)*
                #(#constants: f64,)*
                direction_u: &[f64; #dimension],
                direction_v: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] #fourth_body
        }
    } else {
        quote!()
    };

    let full_function = if emissions.full {
        let third_full_name = format_ident!("{}_third_full", name);
        let fourth_full_name = format_ident!("{}_fourth_full", name);
        let third_full_body = rust_dense_taylor_uncontracted_body(
            &primaries,
            &constant_names,
            &leaves,
            &statements,
            &result,
            3,
        )?;
        let fourth_full_body = rust_dense_taylor_uncontracted_body(
            &primaries,
            &constant_names,
            &leaves,
            &statements,
            &result,
            4,
        )?;
        let third_full_primaries = primary_parameters(&primaries, &quote!(#third_full_body));
        let fourth_full_primaries = primary_parameters(&primaries, &quote!(#fourth_full_body));
        quote! {
            #[inline(always)]
            #visibility fn #third_full_name(
                #(#third_full_primaries: f64,)*
                #(#constants: f64),*
            ) -> [[[f64; #dimension]; #dimension]; #dimension] #third_full_body

            #[inline(always)]
            #visibility fn #fourth_full_name(
                #(#fourth_full_primaries: f64,)*
                #(#constants: f64),*
            ) -> [[[[f64; #dimension]; #dimension]; #dimension]; #dimension]
                #fourth_full_body
        }
    } else {
        quote!()
    };

    let scalar_witness_function = if emissions.witnesses {
        let scalar_witness_dependencies = witness_dependencies(&statements, &witnesses);
        let scalar_witness_scalar_dependencies =
            witness_scalar_dependencies(&statements, &scalar_witness_dependencies)?;
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
        let scalar_witness_statements = scalar_witness_statements.collect::<Vec<_>>();
        let scalar_witness_values = witnesses.iter();
        let scalar_witness_body = quote! {
            #(#scalar_witness_statements)*
            [#(#scalar_witness_values),*]
        };
        let scalar_witness_dependent = primaries
            .iter()
            .filter(|primary| scalar_witness_dependencies.contains(&primary.to_string()))
            .cloned()
            .collect::<Vec<_>>();
        let scalar_witness_primaries =
            primary_parameters(&scalar_witness_dependent, &scalar_witness_body);
        let scalar_witness_constants = constants
            .iter()
            .filter(|constant| scalar_witness_scalar_dependencies.contains(&constant.to_string()));
        quote! {
            #[inline(always)]
            #visibility fn #scalar_witness_name(
                #(#scalar_witness_primaries: f64,)*
                #(#scalar_witness_constants: f64),*
            ) -> [f64; #witness_count] {
                #scalar_witness_body
            }
        }
    } else {
        quote!()
    };

    let cuda_constant = if emissions.cuda {
        let cuda = cuda_source(
            &name,
            &primaries,
            &constant_names,
            &sign_names,
            &leaves,
            &statements,
            &result,
        )?;
        let cuda_literal = Literal::string(&cuda);
        let cuda_name = format_ident!("{}_CUDA_VGH", name.to_string().to_uppercase());
        quote!(#visibility const #cuda_name: &str = #cuda_literal;)
    } else {
        quote!()
    };

    Ok(quote! {
        #generic_function
        #runtime_function
        #order2_function
        #third_function
        #fourth_function
        #full_function
        #scalar_witness_function
        #cuda_constant
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

    fn emitted_item_names(input: TokenStream2) -> Vec<String> {
        let input = syn::parse2::<Input>(input).expect("parse row program");
        let expanded = expand(input).expect("expand row program");
        let file = syn::parse2::<syn::File>(expanded).expect("parse macro expansion");
        file.items
            .into_iter()
            .map(|item| match item {
                syn::Item::Fn(item) => item.sig.ident.to_string(),
                syn::Item::Const(item) => item.ident.to_string(),
                _ => panic!("unexpected emitted row-program item"),
            })
            .collect()
    }

    fn parse_error(input: TokenStream2) -> String {
        match syn::parse2::<Input>(input) {
            Ok(_) => panic!("row program unexpectedly parsed"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn emits_generic_and_shared_symbolic_rust_cuda_schedules() {
        let input = syn::parse2::<Input>(quote! {
            pub(crate) fn sample(q, g; weight, event, scale)
            emit [generic, runtime, order2, third, fourth, full, witnesses, cuda];
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
        assert!(expanded.contains("JetScalar < __ROW_PROGRAM_DERIVATIVE_DIMENSION >"));
        assert!(expanded.contains("RuntimeJetScalar"));
        assert!(expanded.contains("fn sample_runtime"));
        assert!(expanded.contains("fn sample_order2"));
        assert!(expanded.contains("fn sample_third_contracted"));
        assert!(expanded.contains("fn sample_fourth_contracted"));
        assert!(expanded.contains("fn sample_third_full"));
        assert!(expanded.contains("fn sample_fourth_full"));
        assert!(expanded.contains("direction_u"));
        assert!(expanded.contains("direction_v"));
        assert!(expanded.contains("sqrt_stack"));
        assert!(expanded.contains("log_stack"));
        assert!(expanded.contains("SAMPLE_CUDA_VGH"));
        assert!(expanded.contains("double event_term_v = 0.0"));
        assert!(expanded.contains("if ((in.event > 0.0))"));
        assert!(expanded.contains("d_log(adjusted_v"));
        assert!(!expanded.contains("J2"));
        assert!(!expanded.contains("j2_"));
    }

    #[test]
    fn emits_exactly_the_mandatory_per_program_surfaces() {
        let names = emitted_item_names(quote! {
            fn selective(x; shift)
            emit [runtime, order2, third, fourth];
            leaves { curve => curve_stack => d_curve }
            witnesses [curved];
            {
                let shifted = add_constant(x, shift);
                let curved = compose(curve, shifted);
                return curved;
            }
        });

        assert_eq!(
            names,
            vec![
                "selective_runtime".to_owned(),
                "selective_order2".to_owned(),
                "selective_third_contracted".to_owned(),
                "selective_fourth_contracted".to_owned(),
            ]
        );
    }

    #[test]
    fn emission_surfaces_are_mandatory_nonempty_known_and_unique() {
        let missing = parse_error(quote! {
            fn missing(x;)
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(missing.contains("mandatory `emit [ ... ];`"));

        let empty = parse_error(quote! {
            fn empty(x;)
            emit [];
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(empty.contains("must emit at least one surface"));

        let unknown = parse_error(quote! {
            fn unknown(x;)
            emit [jet];
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(
            unknown.contains(
                "must be one of `generic`, `runtime`, `order2`, `third`, `fourth`, `full`, `witnesses`, or `cuda`"
            )
        );

        let duplicate = parse_error(quote! {
            fn duplicate(x;)
            emit [runtime, runtime];
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(duplicate.contains("duplicate row_program emission surface `runtime`"));
    }

    #[test]
    fn rejects_empty_scalar_witness_surface() {
        let input = syn::parse2::<Input>(quote! {
            fn empty_witnesses(x;)
            emit [witnesses];
            leaves {}
            witnesses [];
            { return x; }
        })
        .expect("parse row program");
        let error = expand(input).expect_err("empty witness surface must be rejected");
        assert!(error.to_string().contains("no declared witnesses"));
    }

    #[test]
    fn runtime_rust_schedule_clones_reusable_bindings_and_uses_runtime_workspace() {
        let rust = emitted_function(
            quote! {
                fn runtime_formula(x, y; take, shift)
                emit [runtime];
                leaves { curve => curve_stack => d_curve }
                witnesses [curved];
                {
                    let sum = add(x, y);
                    let shifted = add_constant(sum, shift);
                    let curved = compose(curve, shifted);
                    let mut out = zero();
                    if (take > 0.0) { out = add(curved, x); }
                    return add(out, curved);
                }
            },
            "runtime_formula_runtime",
        )
        .replace(' ', "");

        for formula in [
            "S:::gam_math::jet_scalar::RuntimeJetScalar<'arena>",
            "letx=(*x).clone();",
            "lety=(*y).clone();",
            "value.add_constant(shift)",
            "S::constant(0.0,__row_program_dimension,__row_program_workspace)",
            "letvalue=shifted.clone();",
            "[curved.value()]",
        ] {
            assert!(
                rust.contains(formula),
                "missing generated runtime formula: {formula}\n{rust}"
            );
        }
    }

    #[test]
    fn rust_order2_formulas_pin_sparse_mul_compose_branch_witness_and_symmetry() {
        let rust = emitted_function(
            quote! {
                fn formulas(x, y; take)
                emit [order2];
                leaves { curve => curve_stack => d_curve }
                witnesses [curved];
                {
                    let product = mul(x, y);
                    let curved = compose(curve, product);
                    let mut out = x;
                    if (take > 0.0) { out = add(curved, y); }
                    return out;
                }
            },
            "formulas_order2",
        )
        .replace(' ', "");

        for formula in [
            "fnformulas_order2(x:f64,y:f64,take:f64)",
            "letproduct_g0:f64=y;",
            "letproduct_h0_1:f64=1.0;",
            "letproduct_g1:f64=x;",
            "letcurved_stack0=curve_stack(product_v);",
            "letcurved_g0:f64=(product_g0*curved_stack0[1]);",
            "letcurved_stack0_c0:f64=(curved_stack0[2]*product_g0);",
            "letcurved_stack0_c1:f64=(curved_stack0[2]*product_g1);",
            "letcurved_h0_0:f64=(curved_stack0_c0*product_g0);",
            "letcurved_h0_1:f64=((product_h0_1*curved_stack0[1])+(curved_stack0_c0*product_g1));",
            "letcurved_g1:f64=(product_g1*curved_stack0[1]);",
            "letcurved_h1_1:f64=(curved_stack0_c1*product_g1);",
            "letmutout_g0:f64=1.0;",
            "letmutout_h0_1:f64=0.0;",
            "if(take>0.0){",
            "out_g1=(curved_g1+1.0);",
            "[__row_program_h0_0,__row_program_h0_1],",
            "[__row_program_h0_1,__row_program_h1_1],",
            "[curved_v]",
        ] {
            assert!(
                rust.contains(formula),
                "missing generated formula: {formula}"
            );
        }
        assert!(!rust.contains("JetScalar"));
        assert!(!rust.contains("SparseOrder2"));
        assert!(!rust.contains("*0.0"));
        assert!(!rust.contains("0.0*"));
    }

    /// #932: a gate assigns the channels its term reaches, and no others.
    ///
    /// The union support tells a gate which channels the mutable carries across
    /// ALL of its assignments, and the emitter used to write every one of them
    /// in every gate. A term that reaches three channels then restated the other
    /// forty — as `channel = channel` for a channel an earlier term had set, and
    /// as `channel = 0.0` for one nothing had touched. On the survival
    /// location-scale row that was two thirds of the emitted order-2 body.
    /// Neither form can change a value, so dropping them leaves the schedule
    /// computing exactly what it computed.
    #[test]
    fn a_gate_assigns_only_the_channels_its_term_reaches_932() {
        let rust = emitted_function(
            quote! {
                fn two_terms(x, y; take_a, take_b)
                emit [order2];
                leaves { curve => curve_stack => d_curve }
                witnesses [];
                {
                    let mut out = zero();
                    if (take_a > 0.0) { out = compose(curve, x); }
                    if (take_b > 0.0) { out = add(out, mul(y, y)); }
                    return out;
                }
            },
            "two_terms_order2",
        )
        .replace(' ', "");

        // Each channel the mutable carries is declared zero once, where the
        // scheduler places its declaration.
        for channel in ["out_v", "out_g0", "out_h0_0", "out_g1", "out_h1_1"] {
            assert!(
                rust.contains(&format!("letmut{channel}:f64=0.0;")),
                "the mutable's {channel} channel is declared zero:\n{rust}"
            );
        }

        // The first term is a function of `x` alone, so it must not write the
        // `y` channels — on the released emitter it wrote them as `= 0.0`.
        for restated in ["out_g1=0.0;", "out_h1_1=0.0;", "out_h0_1=0.0;"] {
            assert!(
                !rust.contains(restated),
                "a gate restated a channel that is already zero: {restated}\n{rust}"
            );
        }
        // The second term adds `y·y`, which reaches neither `out_g0` nor
        // `out_h0_0`, so it must not restate them — on the released emitter it
        // wrote them as `channel = channel`.
        for restated in ["out_g0=out_g0;", "out_h0_0=out_h0_0;"] {
            assert!(
                !rust.contains(restated),
                "a gate restated a channel its term leaves alone: {restated}\n{rust}"
            );
        }

        // NON-VACUITY. Each gate still assigns what its term does reach, and
        // the one channel both terms carry is still read-modify-written, or the
        // assertions above would pass on an emitter that assigned nothing.
        for assigned in [
            "out_g0=out_stack0[1];",
            "out_g1=(y+y);",
            "out_h1_1=(1.0+1.0);",
            "out_v=(out_v+(y*y));",
        ] {
            assert!(
                rust.contains(assigned),
                "the gates must still assign the channels their terms reach: {assigned}\n{rust}"
            );
        }
        assert_eq!(
            rust.matches("if(take_a>0.0){").count(),
            1,
            "the first gate is emitted"
        );
        assert_eq!(
            rust.matches("if(take_b>0.0){").count(),
            1,
            "the second gate is emitted"
        );
    }

    #[test]
    fn contracted_formulas_are_direct_sparse_scalar_schedules() {
        let input = quote! {
            fn directional(x, y; take)
            emit [third, fourth];
            leaves { curve => curve_stack => d_curve }
            witnesses [];
            {
                let product = mul(x, y);
                let curved = compose(curve, product);
                let mut out = x;
                if (take > 0.0) { out = add(curved, y); }
                return out;
            }
        };
        let third =
            emitted_function(input.clone(), "directional_third_contracted").replace(' ', "");
        let fourth = emitted_function(input, "directional_fourth_contracted").replace(' ', "");

        for formula in [
            "fndirectional_third_contracted(x:f64,y:f64,take:f64,direction_u:&[f64;2usize],)",
            "let__row_program_product_dense_tmp0_c6:f64=1.0;",
            "let__row_program_curved_dense_stack1=curve_stack(product_c0);",
            "__row_program_curved_dense_stack1[3]",
            "letdense_derivatives=[",
            "dense_derivatives[offset]*direction_u[0]",
            "if(take>0.0){",
        ] {
            assert!(
                third.contains(formula),
                "missing generated third-order formula: {formula}\n{third}"
            );
        }
        for formula in [
            "fndirectional_fourth_contracted(x:f64,y:f64,take:f64,direction_u:&[f64;2usize],direction_v:&[f64;2usize],)",
            "__row_program_curved_dense_stack1[4]",
            "letdirection_01=direction_u[0]*direction_v[1]+direction_u[1]*direction_v[0];",
            "dense_derivatives[offset+1]*direction_01",
        ] {
            assert!(
                fourth.contains(formula),
                "missing generated fourth-order formula: {formula}\n{fourth}"
            );
        }
        for rust in [&third, &fourth] {
            assert!(!rust.contains("JetScalar"));
            assert!(!rust.contains("SparseOrder2"));
            assert!(!rust.contains("*0.0)"));
            assert!(!rust.contains("0.0*"));
        }
    }

    /// The `1/k!` a composition on a primary introduces and the `k!` a
    /// derivative extraction removes cancel in the emitter. On the
    /// third-order surfaces (the root composition specialised, the inner
    /// jet's derivatives read directly) no constant multiply is emitted at
    /// all; on the fourth-order surfaces the dense composition keeps its
    /// Faà di Bruno multiplicities (a `2` for a repeated partition), but the
    /// extracted derivatives are the coefficients themselves. The rigid
    /// Bernoulli row's fourth channel paid seventeen round-trip multiplies
    /// before this, the margin by which its hand kernel won on one host
    /// (#932).
    #[test]
    fn dense_taylor_factorials_cancel_in_the_emitter() {
        let input = quote! {
            fn rigid(eta, slope; q, q1, q2, q3, q4, scale_of, sign: sign)
            emit [third, fourth, full];
            leaves {
                link => supplied,
                observed => observed_stack => d_observed,
                probit => probit_stack => d_probit
            }
            witnesses [];
            {
                let qv = compose(link, eta, q, q1, q2, q3, q4);
                let slope_scaled = scale(slope, scale_of);
                let scale_value = compose(observed, slope_scaled);
                let latent = add(mul(qv, scale_value), slope_scaled);
                let margin = scale(latent, sign);
                return compose(probit, margin);
            }
        };
        let literals = [
            "* 0.5", "* 2.0", "* 3.0", "* 4.0", "* 6.0", "* 8.0", "* 12.0", "* 24.0", "* 0.75",
            "* 0.1666", "* 0.0416", "* 0.3333", "* 0.25", "* 0.125",
        ];
        for surface in ["rigid_third_contracted", "rigid_third_full"] {
            let rust = emitted_function(input.clone(), surface);
            for literal in literals {
                assert!(
                    !rust.contains(literal),
                    "{surface} carries a constant multiply: {literal}\n{rust}"
                );
            }
        }
        for surface in ["rigid_fourth_contracted", "rigid_fourth_full"] {
            let rust = emitted_function(input.clone(), surface);
            // The inner jet (through `margin`) carries no constant.
            let inner = &rust[..rust.find("__row_program_result_dense_stack").expect("root stack")];
            for literal in literals {
                assert!(
                    !inner.contains(literal),
                    "{surface}'s inner jet carries a constant multiply: {literal}\n{rust}"
                );
            }
            // The extracted derivatives are coefficient names: no multiply.
            let start = rust.find("let dense_derivatives = [").or_else(|| rust.find("let derivative = ["));
            let start = start.expect("the extracted derivatives");
            let array = &rust[start..rust[start..].find(']').expect("the array closes") + start];
            assert!(!array.contains('*'), "{surface} scales an extracted derivative:\n{array}\n{rust}");
            // A ratio between orders multiplies the input sum, never the
            // product with the leaf's stack entry.
            assert!(
                !rust.contains("]) * 2.0") && !rust.contains("]) * 0.5"),
                "{surface} scales a post-call product:\n{rust}"
            );
        }
    }

    #[test]
    fn rejects_primary_dependent_runtime_branch() {
        let input = syn::parse2::<Input>(quote! {
            fn bad(q; event)
            emit [generic];
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
            emit [cuda];
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
            // The curvature coefficient `f''·g_a` is hoisted once per axis, so
            // every Hessian entry is one multiply.
            "double curved_stack0_c0 = (curved_stack0[2] * product_g0);",
            "double curved_stack0_c1 = (curved_stack0[2] * product_g1);",
            "double curved_g0 = (product_g0 * curved_stack0[1]);",
            "double curved_h0_0 = (curved_stack0_c0 * product_g0);",
            "double curved_h0_1 = ((product_h0_1 * curved_stack0[1]) + (curved_stack0_c0 * product_g1));",
            "double curved_g1 = (product_g1 * curved_stack0[1]);",
            "double curved_h1_1 = (curved_stack0_c1 * product_g1);",
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

    /// A supplied stack composes without a call: the five entries ARE the
    /// stack (#932). The rigid marginal-link stack and the location-scale
    /// residual stacks are evaluated by their kernel builders at the point the
    /// program recomputes; before this leaf kind each went through a function
    /// whose only use of the point was a NaN select.
    #[test]
    fn supplied_stack_composes_without_a_call() {
        let program = quote! {
            fn given(x; a, b, c, d, e)
            emit [order2, cuda];
            leaves { stack => supplied }
            witnesses [];
            {
                let out = compose(stack, x, a, b, c, d, e);
                return out;
            }
        };
        let rust = emitted_function(program.clone(), "given_order2");
        let compact = rust.replace(' ', "");
        assert!(compact.contains("letout_stack0=[a,b,c,d,e];"), "{rust}");
        assert!(!rust.contains("supplied"), "{rust}");
        // The point is only a differentiation axis here: its value is never
        // read, and the parameter says so instead of tripping the unused
        // lint under `deny(warnings)`. Callers stay positional.
        assert!(compact.contains("fngiven_order2(_x:f64,"), "{rust}");
        let cuda = emitted_cuda(program);
        // CUDA reads program constants through the row-input struct.
        assert!(cuda.contains("double out_stack0[3] = {in.a, in.b, in.c};"), "{cuda}");
        assert!(!cuda.contains("supplied"), "{cuda}");

        let short = syn::parse2::<Input>(quote! {
            fn given(x; a, b)
            emit [order2];
            leaves { stack => supplied }
            witnesses [];
            {
                let out = compose(stack, x, a, b);
                return out;
            }
        })
        .expect("parse row program");
        let error = expand(short).expect_err("a two-entry supplied stack is refused");
        assert!(error.to_string().contains("exactly five entries"), "{error}");
    }

    /// Every definition sinks to its first use, so a statement's derivative
    /// channels are computed AFTER the leaf call its value feeds, not before
    /// it and held live across it (#932: eight spills around the rigid
    /// Bernoulli row's probit call).
    #[test]
    fn definitions_ready_before_a_call_run_are_issued_before_it() {
        let program = quote! {
            fn sunk(x, y; w)
            emit [order2, cuda];
            leaves { probit => probit_stack => d_probit }
            witnesses [];
            {
                let a = mul(x, y);
                let m = compose(probit, a, w);
                return m;
            }
        };
        // `a`'s channels are computable before the call and read after it:
        // they are issued before it, where the hand kernel would write them.
        let rust = emitted_function(program.clone(), "sunk_order2");
        let call = rust.find("probit_stack (a_v").expect("the leaf call");
        let gradient = rust.find("let a_g0").expect("the gradient of `a`");
        let hessian = rust.find("let a_h0_1").expect("the Hessian of `a`");
        assert!(gradient < call && hessian < call, "{rust}");
        let cuda = emitted_cuda(program);
        let call = cuda.find("d_probit(a_v").expect("the CUDA leaf call");
        let gradient = cuda.find("double a_g0").expect("the CUDA gradient of `a`");
        assert!(gradient < call, "{cuda}");

        // Two leaf calls with independent inputs are issued back to back, in
        // program order: the value that reads the first call's result depends
        // on the run and follows the second call, as every channel of it
        // does, so the calls stay adjacent.
        let program = quote! {
            fn adjacent(x, y, z;)
            emit [order2];
            leaves { exponential => exp_stack => d_exp }
            witnesses [];
            {
                let a = compose(exponential, x);
                let b = mul(a, y);
                let c = compose(exponential, z);
                return add(b, c);
            }
        };
        let rust = emitted_function(program, "adjacent_order2");
        let first = rust.find("exp_stack (x)").expect("the first leaf call");
        let second = rust.find("exp_stack (z)").expect("the second leaf call");
        let value = rust.find("let b_v").expect("the value reading the first call");
        let derivative = rust.find("let b_g0").expect("the derivative of b");
        assert!(first < second && second < value && value < derivative, "{rust}");
        let compact = rust.replace(' ', "");
        assert!(
            compact.contains("leta_stack0=exp_stack(x);letc_stack1=exp_stack(z);"),
            "two independent calls are issued back to back:\n{rust}"
        );

        // A call whose point reads an earlier call's result does not join
        // that call's run: everything the first result feeds that the second
        // call does not is issued between them, as the rigid Bernoulli row's
        // index channels are between its observed-scale and probit leaves.
        let program = quote! {
            fn chained(x, y;)
            emit [order2];
            leaves { exponential => exp_stack => d_exp }
            witnesses [];
            {
                let a = compose(exponential, x);
                let b = mul(a, y);
                let c = compose(exponential, b);
                return mul(c, b);
            }
        };
        let rust = emitted_function(program, "chained_order2");
        let second = rust.find("exp_stack (b_v").expect("the dependent leaf call");
        let gradient = rust.find("let b_g0").expect("the gradient of b");
        let hessian = rust.find("let b_h0_1").expect("the Hessian of b");
        assert!(gradient < second && hessian < second, "{rust}");
    }

    /// A leaf call and the definitions that only one gate reads are emitted
    /// at the top of that gate's block, so a row on which the gate is closed
    /// never evaluates them; a call two gates read stays before the first gate
    /// (#932: the location-scale row's entry `exp` under the entry gate, its
    /// event index jet under the event gate).
    #[test]
    fn gate_exclusive_work_is_emitted_inside_its_gate() {
        let program = quote! {
            fn gated(x, y; a, b)
            emit [order2, cuda];
            leaves { exponential => exp_stack => d_exp }
            witnesses [];
            {
                let shared = compose(exponential, y);
                let only = compose(exponential, x);
                let inner = mul(only, shared);
                let later = mul(shared, y);
                let mut out = shared;
                if (a != 0.0) { out = add(shared, inner); }
                if (b != 0.0) { out = add(out, later); }
                return out;
            }
        };
        let rust = emitted_function(program.clone(), "gated_order2");
        let shared_call = rust.find("exp_stack (y)").expect("the shared leaf call");
        let only_call = rust.find("exp_stack (x)").expect("the exclusive leaf call");
        let first_gate = rust.find("if (a != 0.0)").expect("the first gate");
        let first_close = rust[first_gate..].find('}').expect("the first gate closes") + first_gate;
        let second_gate = rust.find("if (b != 0.0)").expect("the second gate");
        assert!(shared_call < first_gate, "{rust}");
        assert!(first_gate < only_call && only_call < first_close, "{rust}");
        let inner = rust.find("let inner_g0").expect("the exclusive index jet");
        assert!(first_gate < inner && inner < first_close, "{rust}");
        let later = rust.find("let later_g1").expect("the second gate's exclusive work");
        let second_close = rust[second_gate..].find('}').expect("the second gate closes") + second_gate;
        assert!(second_gate < later && later < second_close, "{rust}");
        // The exclusive call is issued first inside its block, before the
        // block's own arithmetic.
        let assignment = rust[first_gate..].find("out_v =").expect("the gate's assignment") + first_gate;
        assert!(only_call < assignment, "{rust}");

        let cuda = emitted_cuda(program);
        let shared_call = cuda.find("d_exp(y,").expect("the shared CUDA call");
        let only_call = cuda.find("d_exp(x,").expect("the exclusive CUDA call");
        let first_gate = cuda.find("if ((in.a != 0.0))").expect("the first CUDA gate");
        let first_close = cuda[first_gate..].find("\n    }").expect("the first CUDA gate closes") + first_gate;
        assert!(shared_call < first_gate, "{cuda}");
        assert!(first_gate < only_call && only_call < first_close, "{cuda}");
        let inner = cuda.find("double inner_g0").expect("the exclusive CUDA index jet");
        assert!(first_gate < inner && inner < first_close, "{cuda}");
    }

    /// The composition point of a supplied stack is never read (the stack was
    /// evaluated there by the kernel builder), so a local that only feeds such
    /// a compose has a dead value channel: it is not emitted, while its
    /// derivative channels, which the chain rule reads, are.
    #[test]
    fn a_dead_value_channel_is_not_emitted() {
        let program = quote! {
            fn point(x, y; a, b, c, d, e)
            emit [order2, cuda];
            leaves { stack => supplied }
            witnesses [];
            {
                let p = mul(x, y);
                let out = compose(stack, p, a, b, c, d, e);
                return out;
            }
        };
        let rust = emitted_function(program.clone(), "point_order2");
        assert!(!rust.contains("let p_v"), "{rust}");
        assert!(rust.contains("let p_g0"), "{rust}");
        assert!(rust.contains("let p_h0_1"), "{rust}");
        let cuda = emitted_cuda(program);
        assert!(!cuda.contains("double p_v"), "{cuda}");
        assert!(cuda.contains("double p_g0"), "{cuda}");
    }

    /// A definition that reads a mutable local between two gates that
    /// reassign it is emitted between them, whatever reads it later: sinking
    /// it to its first use would read the second gate's state, and the flush
    /// before a call issued after the second gate would too.
    #[test]
    fn a_mutable_read_stays_between_the_gates_that_reassign_it() {
        let program = quote! {
            fn snapshot(x, y; a, b)
            emit [order2, cuda];
            leaves { exponential => exp_stack => d_exp }
            witnesses [];
            {
                let mut acc = x;
                if (a != 0.0) { acc = mul(acc, y); }
                let between = mul(acc, x);
                if (b != 0.0) { acc = add(acc, x); }
                let late = compose(exponential, y);
                return add(mul(between, late), acc);
            }
        };
        let rust = emitted_function(program.clone(), "snapshot_order2");
        let first_gate = rust.find("if (a != 0.0)").expect("the first gate");
        let first_close = rust[first_gate..].find('}').expect("the first gate closes") + first_gate;
        let second_gate = rust.find("if (b != 0.0)").expect("the second gate");
        for channel in ["let between_v", "let between_g0", "let between_h0_0"] {
            let at = rust.find(channel).expect(channel);
            assert!(first_close < at && at < second_gate, "{channel} is not between the gates:\n{rust}");
        }
        let cuda = emitted_cuda(program);
        let first_gate = cuda.find("if ((in.a != 0.0))").expect("the first CUDA gate");
        let first_close = cuda[first_gate..].find('}').expect("the first CUDA gate closes") + first_gate;
        let second_gate = cuda.find("if ((in.b != 0.0))").expect("the second CUDA gate");
        let at = cuda.find("double between_v").expect("the CUDA value");
        assert!(first_close < at && at < second_gate, "{cuda}");

        // Read only inside a later gate, it is still emitted before the
        // reassigning gate, not at the top of the gate that reads it.
        let program = quote! {
            fn inside(x, y; a, b, c)
            emit [order2];
            leaves { exponential => exp_stack => d_exp }
            witnesses [];
            {
                let mut acc = x;
                if (a != 0.0) { acc = mul(acc, y); }
                let between = mul(acc, x);
                if (b != 0.0) { acc = add(acc, x); }
                let mut out = acc;
                if (c != 0.0) { out = add(out, between); }
                return out;
            }
        };
        let rust = emitted_function(program, "inside_order2");
        let second_gate = rust.find("if (b != 0.0)").expect("the second gate");
        let at = rust.find("let between_v").expect("the value");
        assert!(at < second_gate, "{rust}");
    }

    /// A composition on `scale(x, s)` forms `s·f'` and `s²·f''` once and
    /// reads `x`'s own channels; the scaled local's derivative channels are
    /// never emitted. The rigid Bernoulli row (`compose(probit,
    /// scale(latent_index, outcome_sign))`) paid six sign multiplies per row
    /// for the hand kernel's two before this rule existed (#932).
    #[test]
    fn a_scaled_composition_point_is_absorbed_into_the_outer_stack() {
        let program = quote! {
            fn signed(x, y; s, a)
            emit [order2, cuda];
            leaves { probit => probit_stack => d_probit }
            witnesses [];
            {
                let p = mul(x, y);
                let m = scale(p, s);
                let out = compose(probit, m, a);
                return out;
            }
        };
        // The stack is named after the compose's owner, `out`.
        let rust = emitted_function(program.clone(), "signed_order2");
        assert!(rust.contains("probit_stack (m_v"), "{rust}");
        assert!(rust.contains("let out_stack0_u1 : f64 = (out_stack0 [1] * s)"), "{rust}");
        assert!(rust.contains("let out_stack0_u2 : f64 = ((out_stack0 [2] * s) * s)"), "{rust}");
        assert!(rust.contains("(p_g0 * out_stack0_u1)"), "{rust}");
        assert!(!rust.contains("let m_g0"), "{rust}");
        assert!(!rust.contains("let m_h0_0"), "{rust}");
        // The point, `s·f'`, and `s²·f''` (two factors): four sign multiplies
        // in the whole row, independent of the point's support.
        assert_eq!(rust.matches("* s)").count(), 4, "{rust}");
        let cuda = emitted_cuda(program);
        assert!(cuda.contains("double out_stack0_u1 = (out_stack0[1] * in.s);"), "{cuda}");
        assert!(!cuda.contains("double m_g0"), "{cuda}");

        // A negated point is the same rule with `s = -1`: no multiply at all.
        let program = quote! {
            fn negated(x, y;)
            emit [order2];
            leaves { exponential => exp_stack => d_exp }
            witnesses [];
            {
                let p = mul(x, y);
                let n = neg(p);
                let out = compose(exponential, n);
                return out;
            }
        };
        let rust = emitted_function(program, "negated_order2");
        assert!(rust.contains("exp_stack (n_v"), "{rust}");
        assert!(rust.contains("let out_stack0_u1 : f64 = - (out_stack0 [1])"), "{rust}");
        assert!(rust.contains("let out_stack0_u2 : f64 = out_stack0 [2]"), "{rust}");
        assert!(!rust.contains("let n_g0"), "{rust}");

        // A constant declared `: sign` squares to one: `f''` is read as is,
        // and the row pays two sign multiplies, the point and `s·f'`, as the
        // hand kernel that knows the sign does.
        let program = quote! {
            fn signed_role(x, y; s: sign, a)
            emit [order2, cuda];
            leaves { probit => probit_stack => d_probit }
            witnesses [];
            {
                let p = mul(x, y);
                let m = scale(p, s);
                let out = compose(probit, m, a);
                return out;
            }
        };
        let rust = emitted_function(program.clone(), "signed_role_order2");
        assert!(rust.contains("let out_stack0_u2 : f64 = out_stack0 [2]"), "{rust}");
        assert_eq!(rust.matches("* s)").count(), 2, "{rust}");
        let cuda = emitted_cuda(program);
        assert!(cuda.contains("double out_stack0_u2 = out_stack0[2];"), "{cuda}");

        // A point scaled from a mutable local is not aliased: a gate may
        // reassign the local, and the alias would read the wrong state.
        let program = quote! {
            fn guarded(x, y; s, a)
            emit [order2];
            leaves { probit => probit_stack => d_probit }
            witnesses [];
            {
                let mut p = mul(x, y);
                if (s != 0.0) {
                    p = add(p, x);
                }
                let m = scale(p, s);
                let out = compose(probit, m, a);
                return out;
            }
        };
        let rust = emitted_function(program, "guarded_order2");
        assert!(!rust.contains("out_stack0_u1"), "{rust}");
        assert!(rust.contains("let m_g0"), "{rust}");
    }

    /// An activity gate may be stated on the supplied stack itself (`||` over
    /// its entries), so a caller passes no flag and the program's leaf calls
    /// are not held behind the caller's scans (#932, the location-scale row).
    #[test]
    fn activity_conditions_accept_logical_connectives() {
        let program = quote! {
            fn gated(x; a, b)
            emit [order2, cuda];
            leaves { log => log_stack => d_log }
            witnesses [];
            {
                let mut out = zero();
                if (a != 0.0 || b != 0.0) {
                    out = compose(log, x);
                }
                return out;
            }
        };
        let rust = emitted_function(program.clone(), "gated_order2");
        // Each binary operand is parenthesised by the scalar renderer.
        assert!(rust.contains("(a != 0.0) || (b != 0.0)"), "{rust}");
        let cuda = emitted_cuda(program);
        assert!(cuda.contains("(in.a != 0.0) || (in.b != 0.0)"), "{cuda}");
    }

    #[test]
    fn cuda_compose_temporaries_are_unique_across_repeated_assignments() {
        let cuda = emitted_cuda(quote! {
            fn repeated(q; event)
            emit [order2, cuda];
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

        let rust = emitted_function(
            quote! {
                fn repeated(q; event)
                emit [order2, cuda];
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
            },
            "repeated_order2",
        );
        // `let out_stack0 =` (the leaf call), not `let out_stack0_c0` (the
        // hoisted curvature coefficient that follows it).
        assert_eq!(rust.matches("let out_stack0 =").count(), 1);
        assert_eq!(rust.matches("let out_stack1 =").count(), 1);
        assert_eq!(rust.matches("log_stack (out_v)").count(), 2);
    }

    #[test]
    fn scalar_witness_schedule_is_dependency_sliced_from_the_same_program() {
        let witness = emitted_function(
            quote! {
                fn sliced(q, g; event)
                emit [witnesses];
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
                emit [witnesses];
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

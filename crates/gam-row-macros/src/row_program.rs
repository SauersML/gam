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
        ) => {
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
                "scale" | "add_constant" if arguments.len() == 2 => {
                    let value = parse_program_expr(arguments[0], bindings, constants, leaves)?;
                    validate_scalar(arguments[1], constants)?;
                    if operation == "scale" {
                        Ok(ProgramExpr::Scale(
                            Box::new(value),
                            arguments[1].clone(),
                        ))
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
                    let leaf = leaves.get(&leaf_ident.to_string()).copied().ok_or_else(|| {
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

fn cuda_expression(
    expression: &ProgramExpr,
    owner: &str,
    leaves: &[Leaf],
    constants: &HashSet<String>,
    stack_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<String> {
    let mut child = |expression: &ProgramExpr| {
        cuda_expression(
            expression,
            owner,
            leaves,
            constants,
            stack_index,
            preludes,
        )
    };
    match expression {
        ProgramExpr::Path(ident) => Ok(ident.to_string()),
        ProgramExpr::Zero => Ok("j2_const(0.0)".to_string()),
        ProgramExpr::Scale(value, scalar) => Ok(format!(
            "j2_scale({}, {})",
            child(value)?,
            scalar_cuda(scalar, constants)?
        )),
        ProgramExpr::AddConstant(value, scalar) => Ok(format!(
            "j2_addc({}, {})",
            child(value)?,
            scalar_cuda(scalar, constants)?
        )),
        ProgramExpr::Add(left, right) => {
            Ok(format!("j2_add({}, {})", child(left)?, child(right)?))
        }
        ProgramExpr::Mul(left, right) => {
            Ok(format!("j2_mul({}, {})", child(left)?, child(right)?))
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let suffix = if *stack_index == 0 {
                String::new()
            } else {
                format!("_{stack_index}")
            };
            *stack_index += 1;
            let stack = format!("{owner}_stack{suffix}");
            let cuda_leaf = &leaves[*leaf].cuda;
            let mut leaf_arguments = vec![format!("{value}.v")];
            for argument in arguments {
                leaf_arguments.push(scalar_cuda(argument, constants)?);
            }
            leaf_arguments.push(stack.clone());
            preludes.push(format!(
                "    double {stack}[3];\n    {cuda_leaf}({});",
                leaf_arguments.join(", ")
            ));
            Ok(format!("j2_compose({value}, {stack})"))
        }
    }
}

fn cuda_source(
    name: &Ident,
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
) -> Result<String> {
    let parameters = primaries
        .iter()
        .map(|primary| format!("J2 {primary}"))
        .chain(std::iter::once("const RowIn& in".to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut source = format!(
        "__device__ __forceinline__ J2 {name}(\n        {parameters}) {{\n"
    );
    for statement in statements {
        match statement {
            Statement::Local { name, value, .. } => {
                let mut preludes = Vec::new();
                let mut stack_index = 0;
                let value = cuda_expression(
                    value,
                    &name.to_string(),
                    leaves,
                    constants,
                    &mut stack_index,
                    &mut preludes,
                )?;
                for prelude in preludes {
                    source.push_str(&prelude);
                    source.push('\n');
                }
                source.push_str(&format!("    J2 {name} = {value};\n"));
            }
            Statement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!(
                    "    if ({}) {{\n",
                    scalar_cuda(condition, constants)?
                ));
                for (target, value) in assignments {
                    let mut preludes = Vec::new();
                    let mut stack_index = 0;
                    let value = cuda_expression(
                        value,
                        &target.to_string(),
                        leaves,
                        constants,
                        &mut stack_index,
                        &mut preludes,
                    )?;
                    for prelude in preludes {
                        source.push_str("    ");
                        source.push_str(&prelude);
                        source.push('\n');
                    }
                    source.push_str(&format!("        {target} = {value};\n"));
                }
                source.push_str("    }\n");
            }
        }
    }
    let mut preludes = Vec::new();
    let mut stack_index = 0;
    let result = cuda_expression(
        result,
        "result",
        leaves,
        constants,
        &mut stack_index,
        &mut preludes,
    )?;
    for prelude in preludes {
        source.push_str(&prelude);
        source.push('\n');
    }
    source.push_str(&format!("    return {result};\n}}\n"));
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
                let value =
                    parse_program_expr(&value, &bindings, &constant_names, &leaf_indices)?;
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
    let result = parse_program_expr(
        &body.result,
        &bindings,
        &constant_names,
        &leaf_indices,
    )?;
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
    let cuda_name = format_ident!("{}_CUDA_J2", name.to_string().to_uppercase());

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

        #visibility const #cuda_name: &str = #cuda_literal;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

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
        assert!(expanded.contains("SAMPLE_CUDA_J2"));
        assert!(expanded.contains("J2 event_term = j2_const(0.0)"));
        assert!(expanded.contains("if ((in.event > 0.0))"));
        assert!(expanded.contains("d_log(adjusted.v"));
        assert!(expanded.contains("return j2_add(adjusted, event_term)"));
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
}

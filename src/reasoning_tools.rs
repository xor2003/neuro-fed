use crate::pc_types::PCError;
use std::process::Command;

pub fn sympy_eval(operation: &str, expression: &str) -> Result<String, PCError> {
    let op = operation.trim().to_lowercase();
    let expr = expression.replace('"', "\\\"");
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());

    let script = match op.as_str() {
        "simplify" => format!(
            "import sympy as sp; print(sp.simplify(\"{}\"))",
            expr
        ),
        "expand" => format!("import sympy as sp; print(sp.expand(\"{}\"))", expr),
        "factor" => format!("import sympy as sp; print(sp.factor(\"{}\"))", expr),
        _ => {
            return Err(PCError(format!(
                "Unsupported sympy operation: {}",
                operation
            )))
        }
    };

    let output = Command::new(python)
        .args(["-c", &script])
        .output()
        .map_err(|e| PCError(format!("Failed to spawn python: {}", e)))?;

    if !output.status.success() {
        return Err(PCError(format!(
            "Sympy subprocess failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(feature = "z3-tools")]
pub fn z3_solve_int(var: &str, constraints: &[&str]) -> Result<i64, PCError> {
    let ctx = z3::Context::new(&z3::Config::new());
    let solver = z3::Solver::new(&ctx);
    let symbol = z3::ast::Int::new_const(&ctx, var);

    for c in constraints {
        let trimmed = c.trim();
        if trimmed.contains('%') && trimmed.contains("==") {
            // Example: x % 2 == 0
            let parts: Vec<&str> = trimmed.split("==").collect();
            if parts.len() != 2 {
                return Err(PCError(format!("Unsupported constraint: {}", trimmed)));
            }
            let rhs: i64 = parts[1].trim().parse().map_err(|_| {
                PCError(format!("Invalid modulo RHS in constraint: {}", trimmed))
            })?;
            let left = parts[0].trim();
            let mod_parts: Vec<&str> = left.split('%').collect();
            if mod_parts.len() != 2 {
                return Err(PCError(format!("Unsupported constraint: {}", trimmed)));
            }
            let modulus: i64 = mod_parts[1].trim().parse().map_err(|_| {
                PCError(format!("Invalid modulus in constraint: {}", trimmed))
            })?;
            let modulo = symbol.modulo(&z3::ast::Int::from_i64(&ctx, modulus));
            solver.assert(&modulo._eq(&z3::ast::Int::from_i64(&ctx, rhs)));
            continue;
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() != 3 {
            return Err(PCError(format!("Unsupported constraint: {}", trimmed)));
        }
        let rhs: i64 = tokens[2]
            .parse()
            .map_err(|_| PCError(format!("Invalid RHS in constraint: {}", trimmed)))?;
        let rhs = z3::ast::Int::from_i64(&ctx, rhs);

        match tokens[1] {
            ">" => solver.assert(&symbol.gt(&rhs)),
            ">=" => solver.assert(&symbol.ge(&rhs)),
            "<" => solver.assert(&symbol.lt(&rhs)),
            "<=" => solver.assert(&symbol.le(&rhs)),
            "==" => solver.assert(&symbol._eq(&rhs)),
            _ => return Err(PCError(format!("Unsupported constraint: {}", trimmed))),
        }
    }

    if solver.check() != z3::SatResult::Sat {
        return Err(PCError("Z3 solver returned unsat".to_string()));
    }
    let model = solver.get_model().ok_or_else(|| PCError("No model".to_string()))?;
    let value = model
        .eval(&symbol, true)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| PCError("Failed to extract model value".to_string()))?;
    Ok(value)
}

#[cfg(not(feature = "z3-tools"))]
pub fn z3_solve_int(_var: &str, _constraints: &[&str]) -> Result<i64, PCError> {
    Err(PCError(
        "z3-tools feature disabled (enable with --features z3-tools)".to_string(),
    ))
}

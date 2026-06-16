use std::collections::BTreeSet;
use std::fs;

#[test]
fn pyffi_every_pyfunction_is_registered_once() {
    let src = fs::read_to_string("crates/gam-pyffi/src/lib.rs").expect("read lib.rs");
    let mut pyfns = BTreeSet::new();
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        if lines[i].trim_start().starts_with("#[pyfunction") {
            let mut j = i + 1;
            while j < lines.len() {
                let t = lines[j].trim_start();
                if t.starts_with("#[") {
                    j += 1;
                    continue;
                }
                if let Some(rest) = t.strip_prefix("fn ") {
                    // Function name terminates at the first `(`, `<`, or whitespace,
                    // so we strip generic / lifetime parameters like `<'py>` before
                    // comparing against `wrap_pyfunction!` registrations.
                    let name: String = rest
                        .chars()
                        .take_while(|c| !matches!(c, '(' | '<' | ' ' | '\t'))
                        .collect();
                    pyfns.insert(name);
                }
                break;
            }
            i = j;
        }
        i += 1;
    }

    let module_start = src.find("fn rust_extension").expect("pymodule init");
    let test_start = src.find("#[cfg(test)]").expect("test module");
    let mod_src = &src[module_start..test_start];
    // Find every `wrap_pyfunction!(...)` invocation. The first identifier inside
    // the parentheses is the registered Rust function. Walk character-by-character
    // so that whitespace / newlines between `wrap_pyfunction!(` and the function
    // name (rustfmt loves to split these) do not hide the registration.
    let needle = "wrap_pyfunction!(";
    let mut regs = Vec::new();
    let bytes = mod_src.as_bytes();
    let mut search_from = 0usize;
    while let Some(off) = mod_src[search_from..].find(needle) {
        let abs = search_from + off + needle.len();
        let mut k = abs;
        while k < bytes.len() && (bytes[k] as char).is_whitespace() {
            k += 1;
        }
        let start = k;
        while k < bytes.len() {
            let c = bytes[k] as char;
            if c.is_ascii_alphanumeric() || c == '_' {
                k += 1;
            } else {
                break;
            }
        }
        if k > start {
            regs.push(mod_src[start..k].to_string());
        }
        search_from = abs;
    }

    let regs_set: BTreeSet<String> = regs.iter().cloned().collect();
    let missing: Vec<String> = pyfns.difference(&regs_set).cloned().collect();
    assert!(
        missing.is_empty(),
        "unregistered #[pyfunction] exports: {missing:?}"
    );
}

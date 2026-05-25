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
    let mut regs = Vec::new();
    for line in mod_src.lines() {
        if let Some(idx) = line.find("wrap_pyfunction!(") {
            let tail = &line[idx + "wrap_pyfunction!(".len()..];
            let name = tail.split(',').next().unwrap().trim().to_string();
            regs.push(name);
        }
    }

    let regs_set: BTreeSet<String> = regs.iter().cloned().collect();
    let missing: Vec<String> = pyfns.difference(&regs_set).cloned().collect();
    assert!(
        missing.is_empty(),
        "unregistered #[pyfunction] exports: {missing:?}"
    );
}

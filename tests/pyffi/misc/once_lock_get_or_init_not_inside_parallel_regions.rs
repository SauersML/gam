use std::fs;
use std::path::Path;

#[test]
fn once_lock_get_or_init_not_inside_parallel_regions() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders: Vec<String> = Vec::new();

    let mut stack = vec![src_root];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).expect("read_dir failed");
        for entry in entries {
            let entry = entry.expect("dir entry failed");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let content = fs::read_to_string(&path).expect("read file failed");
            let lines: Vec<&str> = content.lines().collect();
            for i in 0..lines.len() {
                let line = lines[i];
                if line.contains("get_or_init(") {
                    // The risk this gate guards against is a `get_or_init` running
                    // *inside* a rayon parallel closure (first-init landing on a
                    // worker thread). A blind ±N-line window mis-fires when an
                    // unrelated `fn` that merely happens to sit near a parallel
                    // loop also uses a cache. Walk backwards within the SAME
                    // function body (stop at the previous top-level `fn ` opener)
                    // and only flag if a parallel iterator was opened before this
                    // line without an intervening function boundary.
                    let mut opened_parallel = false;
                    for j in (0..i).rev() {
                        let prev = lines[j];
                        // Top-level `fn ` declaration ends the enclosing-scope walk.
                        let trimmed = prev.trim_start();
                        if trimmed.starts_with("fn ")
                            || trimmed.starts_with("pub fn ")
                            || trimmed.starts_with("pub(crate) fn ")
                            || trimmed.starts_with("async fn ")
                        {
                            break;
                        }
                        if prev.contains(".par_iter(") || prev.contains(".into_par_iter(") {
                            opened_parallel = true;
                            break;
                        }
                    }
                    if opened_parallel {
                        offenders.push(format!("{}:{}", path.display(), i + 1));
                    }
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "found get_or_init in proximity of par_iter/into_par_iter:\n{}",
        offenders.join("\n")
    );
}

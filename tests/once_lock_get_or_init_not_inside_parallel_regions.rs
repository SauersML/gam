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
                    let start = i.saturating_sub(80);
                    let end = (i + 80).min(lines.len().saturating_sub(1));
                    let window = &lines[start..=end];
                    if window
                        .iter()
                        .any(|l| l.contains(".par_iter(") || l.contains(".into_par_iter("))
                    {
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

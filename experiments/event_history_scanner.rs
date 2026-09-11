use std::{fs, hint::black_box, time::Instant};

fn ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn old(line: &str, fragment: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    let line_bytes = line.as_bytes();
    let fragment_bytes = fragment.as_bytes();
    if fragment_bytes.is_empty() || line_bytes.len() < fragment_bytes.len() {
        return hits;
    }
    let left = ident(fragment_bytes[0]);
    let right = ident(fragment_bytes[fragment_bytes.len() - 1]);
    let mut start = 0;
    while start + fragment_bytes.len() <= line_bytes.len() {
        if &line_bytes[start..start + fragment_bytes.len()] == fragment_bytes {
            let end = start + fragment_bytes.len();
            if (!left || start == 0 || !ident(line_bytes[start - 1]))
                && (!right || end == line_bytes.len() || !ident(line_bytes[end]))
            {
                hits.push(start);
            }
        }
        start += 1;
    }
    hits
}

fn new(line: &str, fragment: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    let bytes = line.as_bytes();
    if fragment.is_empty() || line.len() < fragment.len() {
        return hits;
    }
    let left = ident(fragment.as_bytes()[0]);
    let right = ident(fragment.as_bytes()[fragment.len() - 1]);
    let mut search = 0;
    while let Some(relative) = line[search..].find(fragment) {
        let start = search + relative;
        let end = start + fragment.len();
        if (!left || start == 0 || !ident(bytes[start - 1]))
            && (!right || end == line.len() || !ident(bytes[end]))
        {
            hits.push(start);
        }
        search = start
            + line[start..]
                .chars()
                .next()
                .expect("nonempty match")
                .len_utf8();
    }
    hits
}

fn words(max: usize) -> Vec<String> {
    let alphabet = ["a", "_", "é", ":", " "];
    let mut all = vec![String::new()];
    let mut level = vec![String::new()];
    for _ in 0..max {
        level = level
            .iter()
            .flat_map(|prefix| {
                alphabet
                    .iter()
                    .map(move |suffix| format!("{prefix}{suffix}"))
            })
            .collect();
        all.extend(level.clone());
    }
    all
}

fn main() {
    let fragments = words(3);
    let lines = words(5);
    let mut comparisons = 0;
    for line in &lines {
        for fragment in &fragments {
            assert_eq!(
                old(line, fragment),
                new(line, fragment),
                "{line:?}, {fragment:?}"
            );
            comparisons += 1;
        }
    }
    let source = fs::read_to_string("build.rs").expect("run from repository root");
    let patterns = [
        "env::var(",
        "todo!",
        "panic!",
        "unwrap()",
        "fn main",
        "#[test]",
        "let _",
        "expect(",
        "== true",
        "unsafe",
        "std::fs",
        "command",
        "read_dir",
        "::",
        "aa",
        "//",
    ];
    let run = |matcher: fn(&str, &str) -> Vec<usize>| {
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..30 {
            for line in source.lines() {
                for pattern in patterns {
                    count += black_box(matcher(black_box(line), black_box(pattern))).len();
                }
            }
        }
        (count, start.elapsed())
    };
    let before = run(old);
    let after = run(new);
    assert_eq!(before.0, after.0);
    eprintln!(
        "{comparisons} exhaustive Unicode/overlap comparisons; equal hit counts {}",
        before.0
    );
    eprintln!(
        "old {:.6}s; new {:.6}s; speedup {:.2}",
        before.1.as_secs_f64(),
        after.1.as_secs_f64(),
        before.1.as_secs_f64() / after.1.as_secs_f64()
    );
}

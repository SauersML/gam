use grep::regex::RegexMatcher;
use grep::searcher::{Searcher, Sink, SinkMatch};
use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::io::{self, Cursor, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use walkdir::WalkDir;

// A custom "Sink" for the grep searcher. It collects all matching lines
// from a single file to build a comprehensive error message.
struct ViolationCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A collector for disallowed `let _ = token;` patterns
struct DisallowedLetCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A collector for tuple destructuring patterns that discard values using `_`
struct TupleWildcardCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A collector for obvious no-op "touch" statements
struct NoopTouchCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A collector for forbidden comment content
struct ForbiddenCommentCollector {
    violations: Vec<String>,
    file_path: PathBuf,
    check_stars_in_doc_comments: bool,
}

// A custom collector for checking if comments have an excessive ratio of uppercase characters
struct CustomUppercaseCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for #[allow(dead_code)] attribute violations
struct DeadCodeCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for #[ignore] test attribute violations
struct IgnoredTestCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for forbidden drop(...) usage in build scripts
struct DropUsageCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

struct EmptyBlockCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

struct DebugAssertCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for forbidden deferred-work comments
struct TodoCommentCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for meaningless conditionals where both branches are identical
struct MeaninglessConditionalCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for `const _: fn() = func_name;` dead-code anchor patterns
struct DeadCodeAnchorCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for #[allow(clippy::no_effect)] with no-op variable access
struct NoEffectCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for "omitted for brevity" comments
struct OmittedForBrevityCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for #[deprecated] attributes
struct DeprecatedCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for placeholder stub functions
struct PlaceholderStubCollector {
    violations: Vec<String>,
    file_path: PathBuf,
    file_content: String,
}

struct DegenerateBooleanCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

static CURRENT_STAGE: OnceLock<Mutex<String>> = OnceLock::new();

fn warnings_enabled() -> bool {
    static ENABLE_WARNINGS: OnceLock<bool> = OnceLock::new();
    *ENABLE_WARNINGS.get_or_init(|| match std::env::var("BUILD_VERBOSE") {
        Ok(value) => {
            let normalized = value.trim();
            normalized.eq_ignore_ascii_case("true")
                || normalized.eq_ignore_ascii_case("yes")
                || normalized == "1"
        }
        Err(_) => false,
    })
}

fn update_stage(label: &str) {
    let tracker = CURRENT_STAGE.get_or_init(|| Mutex::new(String::new()));
    if let Ok(mut guard) = tracker.lock() {
        guard.clear();
        guard.push_str(label);
    }

    if warnings_enabled() {
        println!("cargo:warning=project build stage: {label}");
        io::stdout().flush().ok();
    }
}

fn emit_stage_detail(detail: &str) {
    if warnings_enabled() {
        println!("cargo:warning=project build detail: {detail}");
        io::stdout().flush().ok();
    }
}

fn install_stage_panic_hook() {
    let tracker: &'static Mutex<String> = CURRENT_STAGE.get_or_init(|| Mutex::new(String::new()));
    std::panic::set_hook(Box::new(move |info| {
        let stagename = tracker
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_else(|_| String::from("<stage lock poisoned>"));
        eprintln!("\n⚠️ build script panic while processing stage: {stagename}");
        eprintln!("{info}");
    }));
}

#[allow(clippy::collapsible_if)]
fn detect_total_memory_bytes() -> Option<u64> {
    if let Ok(forced) = std::env::var("GAM_FORCE_TOTAL_MEMORY_BYTES") {
        if let Ok(parsed) = forced.trim().parse::<u64>() {
            return Some(parsed);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let mut parts = rest.split_whitespace();
                    if let Some(rawvalue) = parts.next() {
                        if let Ok(kib) = rawvalue.parse::<u64>() {
                            return Some(kib.saturating_mul(1024));
                        }
                    }
                }
            }
        }
    }

    None
}

fn configure_linker_for_low_memory() {
    if let Ok(value) = std::env::var("GAM_DISABLE_LOW_MEM_WORKAROUND") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "1" | "true" | "yes") {
            return;
        }
    }

    const TEN_GIB: u64 = 10u64 * 1024 * 1024 * 1024;

    match detect_total_memory_bytes() {
        Some(total) if total < TEN_GIB => {
            println!("cargo:rustc-link-arg=-Wl,--no-keep-memory");
            configure_rustc_parallelism_for_low_memory(total);
            if warnings_enabled() {
                println!(
                    "cargo:warning=linker configured for low-memory host (detected {} bytes)",
                    total
                );
            }
        }
        Some(total) => {
            if warnings_enabled() {
                println!(
                    "cargo:warning=total system memory {} bytes >= 10 GiB, using default linker settings",
                    total
                );
            }
        }
        None => {
            if warnings_enabled() {
                println!(
                    "cargo:warning=unable to detect total system memory; using default linker settings"
                );
            }
        }
    }
}

fn configure_rustc_parallelism_for_low_memory(total_memory_bytes: u64) {
    // Build scripts are no longer permitted to emit arbitrary rustc flags; we rely on
    // Cargo profiles (see Cargo.toml) to set codegen-units instead.
    println!(
        "cargo:rustc-env=GAM_LOW_MEMORY_TOTAL_MEMORY_BYTES={}",
        total_memory_bytes
    );
    println!("cargo:rustc-env=GAM_LOW_MEMORY_SERIAL_BUILD=1");
    println!("cargo:rustc-cfg=gam_low_memory_serial_build");
    if warnings_enabled() {
        println!(
            "cargo:warning=low-memory host detected; consider forcing single rustc codegen unit via Cargo profile overrides if builds still fail"
        );
    }
}

impl ViolationCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    // After searching, this method checks if any violations were found.
    // If so, it formats a detailed error message and returns it.
    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} underscore-prefixed variables in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ Underscore-prefixed variable names are not allowed in this project.\n");
        error_msg.push_str(
            "   Either use the variable (removing the underscore) or remove it completely.\n",
        );
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, rename to `_`, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

impl DisallowedLetCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} disallowed 'let _ =' patterns in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Directly ignoring values with 'let _ =' is forbidden in this project.\n",
        );
        error_msg.push_str(
            "   Handle the result explicitly or restructure the code to avoid silent ignores.\n",
        );

        Some(error_msg)
    }
}

impl TupleWildcardCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} tuple destructuring patterns discarding values in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Using '_' placeholders inside tuple destructuring is forbidden in this project.\n",
        );
        error_msg.push_str(
            "   Bind every value explicitly or restructure the code so nothing is silently ignored.\n",
        );

        Some(error_msg)
    }
}

impl NoopTouchCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} no-op touch statements in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ No-op touches (e.g., \"Foo::bar;\" or \".as_ref();\") are forbidden.\n",
        );
        error_msg.push_str("   Remove the dead code or use the value meaningfully.\n");
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

impl ForbiddenCommentCollector {
    fn new(file_path: &Path, check_stars_in_doc_comments: bool) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
            check_stars_in_doc_comments,
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} forbidden comment patterns in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ Comments containing 'DEPRECATED', 'FIXED', 'CRITICAL', 'CORRECTED', 'FIX', 'FIXES', 'NEW', 'CHANGED', 'CHANGES', 'CHANGE', 'MODIFIED', 'MODIFIES', 'MODIFY', 'UPDATED', 'UPDATES', or 'UPDATE' are STRICTLY FORBIDDEN in this project.\n");
        error_msg.push_str("   Simply remove it instead of deprecating something.\n");
        error_msg.push_str("   The '**' pattern is not allowed in regular comments (but is allowed in doc comments).\n");
        error_msg.push_str(
            "   Comments where over 80% of alphabetic characters are uppercase are not allowed.\n",
        );
        error_msg.push_str("   Please remove these patterns before committing.\n");

        Some(error_msg)
    }
}

impl CustomUppercaseCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} comments with excessive uppercase alphabetic characters in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Comments where over 80% of alphabetic characters are uppercase are STRICTLY FORBIDDEN in this project.\n",
        );
        error_msg.push_str("   STRONGLY CONSIDER deleting the comment completely.\n");

        Some(error_msg)
    }
}

impl DeadCodeCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} forbidden 'allow(...)' attributes in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ The following attributes are STRICTLY FORBIDDEN in this project:\n");
        error_msg.push_str("   - #[allow(dead_code)]\n");
        error_msg.push_str("   - #[allow(unused)] / #![allow(unused)]\n");
        error_msg.push_str("   - #[allow(unused_imports)]\n");
        error_msg.push_str("   - #[allow(unused_variables)]\n");
        error_msg.push_str("   - #[allow(clippy::*)] (any clippy lint)\n");
        error_msg.push_str(
            "\n   Do not suppress these warnings. Delete or use the unused code/imports instead.\n",
        );
        error_msg.push_str(
            "   Clippy lints are pointless to suppress — we don't run clippy and don't care about its opinions.\n",
        );
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

impl IgnoredTestCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} #[ignore] test attributes in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ #[ignore] TEST ATTRIBUTES ARE STRICTLY FORBIDDEN IN THIS PROJECT!\n");
        error_msg.push_str("   IGNORING TESTS IS NEVER ALLOWED FOR ANY REASON.\n");
        error_msg.push_str("   Fix the test so it can run properly without being ignored.\n");

        Some(error_msg)
    }
}

impl DropUsageCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} disallowed drop(...) usages in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ Explicit drop(...) calls are forbidden in this project.\n");
        error_msg.push_str("   Restructure the code to let values go out of scope naturally.\n");

        Some(error_msg)
    }
}

impl EmptyBlockCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} empty control-flow blocks in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ Empty control-flow blocks are forbidden in this project.\n");
        error_msg.push_str("   Remove the block or add meaningful logic.\n");
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

impl DebugAssertCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} debug_assert! usages in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ debug_assert! is forbidden in this project.\n");
        error_msg.push_str("   Use assert! instead.\n");

        Some(error_msg)
    }
}

impl TodoCommentCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} TODO/FIXME comments in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ TODO/FIXME/\"for now\" COMMENTS ARE FORBIDDEN IN THIS PROJECT!\n");
        error_msg.push_str("   These comments indicate incomplete work. DO IT NOW!\n");
        error_msg.push_str("   Do NOT just remove the comment - implement the thing first!\n");

        Some(error_msg)
    }
}

impl MeaninglessConditionalCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} meaningless conditionals in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Meaningless conditionals (identical if/else branches) are STRICTLY FORBIDDEN.\n",
        );
        error_msg
            .push_str("   Either remove the conditional entirely or ensure branches differ.\n");
        error_msg.push_str(
            "   If-else with same result is an anti-pattern to make unused variables appear used.\n",
        );

        Some(error_msg)
    }
}

impl DeadCodeAnchorCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} dead-code anchor pattern(s) in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ `const _: fn() = func_name;` is a dead-code preservation hack.\n");
        error_msg.push_str(
            "   This pattern forces the compiler to keep otherwise-unused functions alive.\n",
        );
        error_msg.push_str("   Either wire the code into a real call site or delete it.\n");
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

impl Sink for DeadCodeAnchorCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

impl NoEffectCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} #[allow(clippy::no_effect)] attributes in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ #[allow(clippy::no_effect)] IS STRICTLY FORBIDDEN IN THIS PROJECT!\n");
        error_msg.push_str(
            "   This is used to suppress warnings about no-op variable access like { variablename; }\n",
        );
        error_msg.push_str(
            "   If a parameter is truly unused, delete it and refactor the function signature.\n",
        );
        error_msg.push_str("   Do NOT use no-op statements to pretend variables are being used.\n");
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

impl OmittedForBrevityCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} \"omitted for brevity\" comments in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ \"OMITTED FOR BREVITY\" COMMENTS ARE STRICTLY FORBIDDEN IN THIS PROJECT!\n",
        );
        error_msg.push_str("   These comments hide incomplete implementations or deleted code.\n");
        error_msg
            .push_str("   DO NOT omit anything. Include all code, tests, and implementations.\n");

        Some(error_msg)
    }
}

impl DeprecatedCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} #[deprecated] attributes in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ #[deprecated] ATTRIBUTES ARE STRICTLY FORBIDDEN IN THIS PROJECT!\n");
        error_msg.push_str("   Simply remove it instead of deprecating something.\n");

        Some(error_msg)
    }
}

impl PlaceholderStubCollector {
    fn new(file_path: &Path, content: &str) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
            file_content: content.to_string(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} placeholder stub functions in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ PLACEHOLDER STUB FUNCTIONS ARE STRICTLY FORBIDDEN IN THIS PROJECT!\n");
        error_msg.push_str(
            "   Functions that return Ok(Self { ... }) with all empty vectors/hashmaps are placeholders.\n",
        );
        error_msg
            .push_str("   These exist solely to satisfy the compiler without doing actual work.\n");
        error_msg
            .push_str("   Implement the function properly instead of using empty placeholders.\n");

        Some(error_msg)
    }
}

impl DegenerateBooleanCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let filename = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} degenerate boolean composites in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Tautological, contradictory, or trivially reducible boolean composites are forbidden.\n",
        );
        error_msg.push_str(
            "   These patterns often exist only to fake-use variables or hide dead logic.\n",
        );
        error_msg.push_str("   Rewrite the condition directly or delete it.\n");
        error_msg.push_str(
            "\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n",
        );

        Some(error_msg)
    }
}

// Implement the `Sink` trait for our collector.
// The `matched` method is called by the searcher for every line that matches the regex.
impl Sink for ViolationCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        let mut hasviolation = false;
        let mut token = String::new();
        for ch in line_text.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                token.push(ch);
            } else if !token.is_empty() {
                if is_disallowed_underscore_token(&token) {
                    hasviolation = true;
                    break;
                }
                token.clear();
            }
        }
        if !hasviolation && !token.is_empty() && is_disallowed_underscore_token(&token) {
            hasviolation = true;
        }

        if hasviolation {
            // Format the violation string exactly as the `rg -n` command would.
            self.violations.push(format!("{line_number}:{line_text}"));
        }

        // Return `Ok(true)` to continue searching for more matches in the same file.
        Ok(true)
    }
}

const SIMD_ALLOWLIST: &[&str] = &["_mm_prefetch", "_MM_HINT_T0", "_SC_PAGESIZE"];
const SIMD_PREFIX_ALLOWLIST: &[&str] = &["_mm", "_MM", "__m"];

fn is_allowed_simd_token(token: &str) -> bool {
    SIMD_ALLOWLIST.contains(&token)
        || SIMD_PREFIX_ALLOWLIST
            .iter()
            .any(|prefix| token.starts_with(prefix))
}

fn is_disallowed_underscore_token(token: &str) -> bool {
    if token == "_" {
        return false;
    }
    if is_allowed_simd_token(token) {
        return false;
    }
    token.starts_with('_')
}

impl Sink for DisallowedLetCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

impl Sink for TupleWildcardCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        if tuple_pattern_is_fully_ignored(line_text) {
            self.violations.push(format!("{line_number}:{line_text}"));
        }

        Ok(true)
    }
}

impl Sink for NoopTouchCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        let trimmed = line_text.trim();
        if matches!(trimmed, "break;" | "continue;" | "return;") {
            return Ok(true);
        }
        if trimmed.starts_with("return ") || trimmed.starts_with("return(") {
            return Ok(true);
        }
        if trimmed.starts_with("let ") || trimmed.contains('=') {
            return Ok(true);
        }
        if trimmed
            .chars()
            .find(|c| !c.is_whitespace())
            .is_some_and(|c| !c.is_ascii_alphabetic() && c != '_')
        {
            return Ok(true);
        }
        // Skip stripped string literals: after string stripping, string
        // contents become sequences of 'x' characters.  A line like
        //   "some long string";
        // becomes  xxxxxxxxxxxxxxxx;  which the regex falsely matches.
        {
            let sans_semi = trimmed.trim_end_matches(';').trim();
            if !sans_semi.is_empty() && sans_semi.chars().all(|c| c == 'x') {
                return Ok(true);
            }
        }

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

fn tuple_pattern_is_fully_ignored(line_text: &str) -> bool {
    let Some(pattern) = extract_tuple_pattern(line_text) else {
        return false;
    };

    let components = split_top_level_components(pattern);
    if components.is_empty() {
        return false;
    }

    components.into_iter().all(is_component_ignored)
}

fn extract_tuple_pattern(line_text: &str) -> Option<&str> {
    let let_pos = line_text.find("let")?;
    let after_let = &line_text[let_pos + 3..];
    let paren_start_rel = after_let.find('(')?;
    let paren_start = let_pos + 3 + paren_start_rel;

    let mut depth = 0usize;
    let mut paren_end = None;
    for (offset, ch) in line_text[paren_start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    paren_end = Some(paren_start + offset);
                    break;
                }
            }
            _ => {}
        }
    }

    let end = paren_end?;
    Some(&line_text[paren_start + 1..end])
}

fn split_top_level_components(pattern: &str) -> Vec<&str> {
    let mut components = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;

    for (idx, ch) in pattern.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' if depth > 0 => depth -= 1,
            ')' | ']' | '}' => {}
            ',' if depth == 0 => {
                components.push(pattern[start..idx].trim());
                start = idx + 1;
            }
            _ => {}
        }
    }

    if start <= pattern.len() {
        components.push(pattern[start..].trim());
    }

    components.retain(|component| !component.is_empty());
    components
}

fn is_component_ignored(component: &str) -> bool {
    let trimmed = component.trim();
    if trimmed.is_empty() {
        return true;
    }

    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        let inner = &trimmed[1..trimmed.len() - 1];
        let inner_components = split_top_level_components(inner);
        return !inner_components.is_empty()
            && inner_components.into_iter().all(is_component_ignored);
    }

    if trimmed.contains('@') {
        return false;
    }

    let mut candidate = trimmed;
    loop {
        let stripped = candidate.trim_start();
        if let Some(rest) = stripped.strip_prefix('&') {
            candidate = rest;
            continue;
        }
        if let Some(rest) = stripped.strip_prefix("mut ") {
            candidate = rest;
            continue;
        }
        if let Some(rest) = stripped.strip_prefix("ref ") {
            candidate = rest;
            continue;
        }
        candidate = stripped;
        break;
    }

    let candidate = candidate.trim();
    if candidate.is_empty() {
        return false;
    }

    if candidate == "_" {
        return true;
    }

    if candidate.starts_with('_')
        && candidate
            .chars()
            .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return true;
    }

    false
}

// Implement the Sink trait for the forbidden comment collector
impl Sink for ForbiddenCommentCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Skip ** in doc comments if not checking for them
        // But NEVER skip any line containing FIXED, CORRECTED, or FIX
        if !self.check_stars_in_doc_comments
            && is_doc_comment(line_text)
            && line_text.contains("**")
            && !line_text.contains("FIXED")
            && !line_text.contains("CRITICAL")
            && !line_text.contains("CORRECTED")
            && !line_text.contains("FIX")
            && !line_text.contains("FIXES")
            && !line_text.contains("NEW")
            && !line_text.contains("CHANGED")
            && !line_text.contains("CHANGES")
            && !line_text.contains("CHANGE")
            && !line_text.contains("MODIFIED")
            && !line_text.contains("MODIFIES")
            && !line_text.contains("MODIFY")
            && !line_text.contains("UPDATED")
            && !line_text.contains("UPDATES")
            && !line_text.contains("UPDATE")
        {
            // Skip this match, it's just ** in a doc comment
            return Ok(true);
        }

        // Format the violation string
        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

// Implement the Sink trait for the uppercase character collector
impl Sink for CustomUppercaseCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Check if it's a comment line
        if !line_text.trim_start().starts_with("//")
            && !line_text.contains("/*")
            && !line_text.starts_with("///")
        {
            return Ok(true); // Not a comment, skip
        }

        // Extract just the comment part (remove the // or /* prefix)
        let comment_text = if line_text.trim_start().starts_with("///") {
            line_text.trim_start()[3..].trim()
        } else if line_text.trim_start().starts_with("//") {
            line_text.trim_start()[2..].trim()
        } else if let Some(idx) = line_text.find("/*") {
            match line_text[idx + 2..].find("*/") {
                Some(end) => line_text[idx + 2..idx + 2 + end].trim(),
                None => line_text[idx + 2..].trim(),
            }
        } else {
            return Ok(true); // Not a comment we can parse, skip
        };

        // Find all alphabetic characters and non-whitespace characters for ratio checks.
        let alpha_count = comment_text.chars().filter(|c| c.is_alphabetic()).count();
        let nonwhitespace_count = comment_text.chars().filter(|c| !c.is_whitespace()).count();

        if alpha_count > 0 && nonwhitespace_count > 0 {
            // Only count uppercase letters that are part of multi-letter words.
            let mut uppercase_count = 0usize;
            let mut run: Vec<char> = Vec::new();
            let flush_run = |run: &mut Vec<char>, uppercase_count: &mut usize| {
                if run.len() > 1 {
                    *uppercase_count += run.iter().filter(|c| c.is_uppercase()).count();
                }
                run.clear();
            };

            for ch in comment_text.chars() {
                if ch.is_alphabetic() {
                    run.push(ch);
                } else {
                    flush_run(&mut run, &mut uppercase_count);
                }
            }
            flush_run(&mut run, &mut uppercase_count);

            let uppercase_ratio = uppercase_count as f64 / alpha_count as f64;
            let alpha_ratio = alpha_count as f64 / nonwhitespace_count as f64;

            // Ignore math-heavy or single-letter comments by requiring enough alphabetic content.
            let has_enough_alpha = alpha_count >= 6 && alpha_ratio >= 0.6;

            if uppercase_ratio > 0.8 && has_enough_alpha {
                self.violations.push(format!("{line_number}:{line_text}"));
            }
        }

        Ok(true)
    }
}

impl Sink for DeadCodeCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        // Get the line number and the content of the matched line.
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Format the violation string
        self.violations.push(format!("{line_number}:{line_text}"));

        // Return `Ok(true)` to continue searching for more matches in the same file.
        Ok(true)
    }
}

impl Sink for IgnoredTestCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        // Get the line number and the content of the matched line.
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Format the violation string
        self.violations.push(format!("{line_number}:{line_text}"));

        // Return `Ok(true)` to continue searching for more matches in the same file.
        Ok(true)
    }
}

impl Sink for DropUsageCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        let is_drop_definition = line_text.contains("fn drop")
            || line_text.contains("impl Drop")
            || line_text.contains("trait Drop");

        if is_drop_definition {
            return Ok(true);
        }

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

impl Sink for DebugAssertCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

impl Sink for TodoCommentCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

impl Sink for MeaninglessConditionalCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Skip pure comments
        let trimmed = line_text.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("/*") {
            return Ok(true);
        }

        // Extract the two branches from the if-else expression
        // Pattern: if COND { BRANCH1 } else { BRANCH2 }
        if let Some((branch1, branch2)) = extract_if_else_branches(line_text) {
            let b1 = branch1.trim();
            let b2 = branch2.trim();
            if b1 == b2 && !b1.is_empty() {
                self.violations.push(format!("{line_number}:{line_text}"));
            }
        }

        Ok(true)
    }
}

// Helper function to extract branches from a single-line if-else expression
fn extract_if_else_branches(line: &str) -> Option<(String, String)> {
    // Find "if" followed by condition and two brace-delimited blocks
    let if_pos = line.find("if ")?;
    let rest = &line[if_pos..];

    // Find the first opening brace (start of then-branch)
    let first_open = rest.find('{')?;
    let after_first_open = &rest[first_open + 1..];

    // Find the matching close brace for the then-branch
    // Simple approach: find first } (works for single expressions)
    let first_close = after_first_open.find('}')?;
    let branch1 = &after_first_open[..first_close];

    // Now look for "else {"
    let after_first_block = &after_first_open[first_close + 1..];
    let else_pos = after_first_block.find("else")?;
    let after_else = &after_first_block[else_pos + 4..];

    // Find the opening brace after else
    let second_open = after_else.find('{')?;
    let aftersecond_open = &after_else[second_open + 1..];

    // Find the closing brace
    let second_close = aftersecond_open.find('}')?;
    let branch2 = &aftersecond_open[..second_close];

    Some((branch1.to_string(), branch2.to_string()))
}

impl Sink for NoEffectCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();
        self.violations.push(format!("{line_number}:{line_text}"));
        Ok(true)
    }
}

impl Sink for OmittedForBrevityCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();
        self.violations.push(format!("{line_number}:{line_text}"));
        Ok(true)
    }
}

impl Sink for DeprecatedCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();
        self.violations.push(format!("{line_number}:{line_text}"));
        Ok(true)
    }
}

impl Sink for PlaceholderStubCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Check if this is actually a placeholder stub by examining surrounding context
        let lines: Vec<&str> = self.file_content.lines().collect();
        let line_idx = line_number as usize;
        if line_idx > 0 && line_idx <= lines.len() {
            let start_idx = line_idx - 1;
            let end_idx = (start_idx + 10).min(lines.len());
            let context = lines[start_idx..end_idx].join("\n");

            // Check if this looks like a placeholder with multiple empty collections
            let empty_count = context.matches("vec![]").count()
                + context.matches("Vec::new()").count()
                + context.matches("HashMap::new()").count()
                + context.matches("BTreeMap::new()").count()
                + context.matches("BTreeSet::new()").count()
                + context.matches("HashSet::new()").count();

            // If there are 2+ empty collections, it's likely a placeholder
            if empty_count >= 2 {
                self.violations.push(format!("{line_number}:{line_text}"));
            }
        }

        Ok(true)
    }
}

#[derive(Clone, Debug)]
enum EmptyBlockTokenKind {
    Ident(String),
    OpenBrace,
    CloseBrace,
    Semicolon,
    Arrow,
}

#[derive(Clone, Debug)]
struct EmptyBlockToken {
    kind: EmptyBlockTokenKind,
    line: usize,
    offset: usize,
    depth: usize,
}

fn main() {
    install_stage_panic_hook();
    configure_linker_for_low_memory();

    // Cargo re-runs the build script when any file in the package changes.
    // Do not narrow the rerun scope with cargo:rerun-if-changed directives.
    update_stage("initialization");

    // Emit build timestamp for version command (always, even when lint checks are skipped)
    let build_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("cargo:rustc-env=GAM_BUILD_TIMESTAMP={}", build_time);

    // Capture release tag if provided by CI
    if let Ok(release_tag) = std::env::var("GAM_RELEASE_TAG") {
        println!("cargo:rustc-env=GAM_RELEASE_TAG={}", release_tag);
    }

    // Skip lint checks during release builds or cross-compilation
    // (the grep crate won't be available in target deps during cross-compile)
    if std::env::var("GAM_SKIP_LINT_CHECKS").is_ok() {
        update_stage("skipping lint checks (GAM_SKIP_LINT_CHECKS set)");
        let out_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR not set");
        let lint_path = Path::new(&out_dir).join("lint_errors.rs");
        std::fs::write(&lint_path, "// Lint checks skipped\n").unwrap_or_else(|e| {
            panic!("failed to write lint errors to {}: {e}", lint_path.display());
        });
        return;
    }

    // Manually check for unused variables in the build script
    update_stage("manual lint self-check");
    manually_check_for_unusedvariables();

    // Collect all violations from all checks
    let mut allviolations = Vec::new();

    // Pre-build stripped file contents cache for cross-file reference analysis.
    // Each entry is (path, comment/string-stripped content).  Built once, shared
    // by any scan that needs codebase-wide reference counting.
    update_stage("build stripped file contents cache");
    let file_contents_cache = build_stripped_file_cache();

    // Scan Rust source files for underscore prefixed variables
    update_stage("scan underscore-prefixed bindings");
    let underscoreviolations = scan_for_underscore_prefixes();
    let underscore_report = format!(
        "underscore scan identified {} violation groups",
        underscoreviolations.len()
    );
    emit_stage_detail(&underscore_report);
    allviolations.extend(underscoreviolations);

    // Scan Rust source files for disallowed `let _ = token;` patterns
    update_stage("scan disallowed let ignore patterns");
    let disallowed_letviolations = scan_for_disallowed_let_patterns();
    let disallowed_let_report = format!(
        "disallowed let pattern scan identified {} violation groups",
        disallowed_letviolations.len()
    );
    emit_stage_detail(&disallowed_let_report);
    allviolations.extend(disallowed_letviolations);

    // Scan Rust source files for no-op touch statements
    update_stage("scan no-op touch statements");
    let noop_touchviolations = scan_for_noop_touch_patterns();
    let noop_touch_report = format!(
        "no-op touch scan identified {} violation groups",
        noop_touchviolations.len()
    );
    emit_stage_detail(&noop_touch_report);
    allviolations.extend(noop_touchviolations);

    // Scan Rust source files for tuple destructuring patterns that discard values
    update_stage("scan tuple destructuring ignores");
    let tuplewildcardviolations = scan_for_tuplewildcard_patterns();
    let tuplewildcard_report = format!(
        "tuple destructuring ignore scan identified {} violation groups",
        tuplewildcardviolations.len()
    );
    emit_stage_detail(&tuplewildcard_report);
    allviolations.extend(tuplewildcardviolations);

    // Scan Rust source files for forbidden comment patterns
    update_stage("scan forbidden comment patterns");
    let commentviolations = scan_for_forbidden_comment_patterns();
    let comment_report = format!(
        "forbidden comment scan identified {} violation groups",
        commentviolations.len()
    );
    emit_stage_detail(&comment_report);
    allviolations.extend(commentviolations);

    // Scan Rust source files for #[allow(dead_code)] attributes
    update_stage("scan allow(dead_code) attributes");
    let dead_codeviolations = scan_for_allow_dead_code();
    let dead_code_report = format!(
        "allow(dead_code) scan identified {} violation groups",
        dead_codeviolations.len()
    );
    emit_stage_detail(&dead_code_report);
    allviolations.extend(dead_codeviolations);

    // Scan Rust source files for #[ignore] test attributes
    update_stage("scan #[ignore] test annotations");
    let ignored_testviolations = scan_for_ignored_tests();
    let ignored_report = format!(
        "ignored test scan identified {} violation groups",
        ignored_testviolations.len()
    );
    emit_stage_detail(&ignored_report);
    allviolations.extend(ignored_testviolations);

    update_stage("scan newly-added #[cfg(test)] production gates");
    let cfg_test_evasionviolations = scan_for_cfg_test_evasion();
    let cfg_test_evasion_report = format!(
        "cfg(test) evasion scan identified {} violation groups",
        cfg_test_evasionviolations.len()
    );
    emit_stage_detail(&cfg_test_evasion_report);
    allviolations.extend(cfg_test_evasionviolations);

    // Scan build scripts for forbidden drop(...) usage
    update_stage("scan build script drop usage");
    let drop_usageviolations = scan_for_drop_in_build_scripts();
    let drop_usage_report = format!(
        "build script drop scan identified {} violation groups",
        drop_usageviolations.len()
    );
    emit_stage_detail(&drop_usage_report);
    allviolations.extend(drop_usageviolations);

    // Scan Rust source files for forbidden drop(...) usage
    update_stage("scan drop usage");
    let drop_usageviolations = scan_for_drop_usage();
    let drop_usage_report = format!(
        "drop usage scan identified {} violation groups",
        drop_usageviolations.len()
    );
    emit_stage_detail(&drop_usage_report);
    allviolations.extend(drop_usageviolations);

    update_stage("scan empty control-flow blocks");
    let empty_blockviolations = scan_for_empty_control_blocks();
    let empty_block_report = format!(
        "empty control-flow block scan identified {} violation groups",
        empty_blockviolations.len()
    );
    emit_stage_detail(&empty_block_report);
    allviolations.extend(empty_blockviolations);

    update_stage("scan debug_assert usage");
    let debug_assertviolations = scan_for_debug_assert_usage();
    let debug_assert_report = format!(
        "debug_assert scan identified {} violation groups",
        debug_assertviolations.len()
    );
    emit_stage_detail(&debug_assert_report);
    allviolations.extend(debug_assertviolations);

    // Scan for deferred-work comments
    update_stage("scan TODO/FIXME comments");
    let todoviolations = scan_for_todo_comments();
    let todo_report = format!(
        "TODO/FIXME comment scan identified {} violation groups",
        todoviolations.len()
    );
    emit_stage_detail(&todo_report);
    allviolations.extend(todoviolations);

    // Scan for meaningless conditionals (same branches)
    update_stage("scan meaningless conditionals");
    let meaningless_condviolations = scan_for_meaningless_conditionals();
    let meaningless_cond_report = format!(
        "meaningless conditional scan identified {} violation groups",
        meaningless_condviolations.len()
    );
    emit_stage_detail(&meaningless_cond_report);
    allviolations.extend(meaningless_condviolations);

    // Guard: Cargo.toml must keep warnings = "deny" under [lints.rust].
    update_stage("check Cargo.toml lint level");
    if let Ok(cargo_toml) = std::fs::read_to_string("Cargo.toml") {
        let in_lints_rust = cargo_toml.contains("[lints.rust]");
        let has_deny = cargo_toml.lines().any(|l| {
            let t = l.trim();
            t.starts_with("warnings") && t.contains('=') && t.contains("\"deny\"")
        });
        if in_lints_rust && !has_deny {
            allviolations.push(
                "\n\u{274c} ERROR: Cargo.toml [lints.rust] must have warnings = \"deny\".\n   \
                 Downgrading to \"warn\" disables the safety net. Revert this change."
                    .to_string(),
            );
        }
    }

    // Scan for fake usage patterns (dummy checks masking unused variables)
    update_stage("scan fake usage patterns");
    let fake_usageviolations = scan_for_fake_usage();
    let fake_usage_report = format!(
        "fake usage scan identified {} violation groups",
        fake_usageviolations.len()
    );
    emit_stage_detail(&fake_usage_report);
    allviolations.extend(fake_usageviolations);

    update_stage("scan inert touch calls");
    let inert_touchviolations = scan_for_inert_touch_calls();
    let inert_touch_report = format!(
        "inert touch scan identified {} violation groups",
        inert_touchviolations.len()
    );
    emit_stage_detail(&inert_touch_report);
    allviolations.extend(inert_touchviolations);

    update_stage("scan suspicious macro wrappers");
    let suspicious_macroviolations = scan_for_suspicious_macro_wrappers();
    let suspicious_macro_report = format!(
        "suspicious macro scan identified {} violation groups",
        suspicious_macroviolations.len()
    );
    emit_stage_detail(&suspicious_macro_report);
    allviolations.extend(suspicious_macroviolations);

    update_stage("scan assertion alias evasions");
    let assertion_aliasviolations = scan_for_assertion_alias_evasions();
    let assertion_alias_report = format!(
        "assertion alias scan identified {} violation groups",
        assertion_aliasviolations.len()
    );
    emit_stage_detail(&assertion_alias_report);
    allviolations.extend(assertion_aliasviolations);

    update_stage("scan degenerate boolean composites");
    let degenerate_booleanviolations = scan_for_degenerate_boolean_expressions();
    let degenerate_boolean_report = format!(
        "degenerate boolean scan identified {} violation groups",
        degenerate_booleanviolations.len()
    );
    emit_stage_detail(&degenerate_boolean_report);
    allviolations.extend(degenerate_booleanviolations);

    // Scan for `const _: fn() = func;` dead-code anchor hacks
    update_stage("scan dead-code anchor patterns");
    let dead_code_anchorviolations = scan_for_dead_code_anchors();
    let dead_code_anchor_report = format!(
        "dead-code anchor scan identified {} violation groups",
        dead_code_anchorviolations.len()
    );
    emit_stage_detail(&dead_code_anchor_report);
    allviolations.extend(dead_code_anchorviolations);

    // Scan for #[allow(clippy::no_effect)] attributes
    update_stage("scan #[allow(clippy::no_effect)] attributes");
    let no_effectviolations = scan_for_no_effect_allow();
    let no_effect_report = format!(
        "no_effect allow scan identified {} violation groups",
        no_effectviolations.len()
    );
    emit_stage_detail(&no_effect_report);
    allviolations.extend(no_effectviolations);

    // Scan for "omitted for brevity" and similar comments
    update_stage("scan omitted for brevity comments");
    let omittedviolations = scan_for_omitted_for_brevity();
    let omitted_report = format!(
        "omitted for brevity scan identified {} violation groups",
        omittedviolations.len()
    );
    emit_stage_detail(&omitted_report);
    allviolations.extend(omittedviolations);

    // Scan for #[deprecated] attributes
    update_stage("scan #[deprecated] attributes");
    let deprecatedviolations = scan_for_deprecated();
    let deprecated_report = format!(
        "deprecated attribute scan identified {} violation groups",
        deprecatedviolations.len()
    );
    emit_stage_detail(&deprecated_report);
    allviolations.extend(deprecatedviolations);

    // Scan for placeholder stub functions
    update_stage("scan placeholder stub functions");
    let placeholderviolations = scan_for_placeholder_stubs();
    let placeholder_report = format!(
        "placeholder stub scan identified {} violation groups",
        placeholderviolations.len()
    );
    emit_stage_detail(&placeholder_report);
    allviolations.extend(placeholderviolations);

    // Scan for dead public items (pub but never referenced anywhere)
    update_stage("scan dead public items");
    let dead_pubviolations = scan_for_dead_public_items(&file_contents_cache);
    let dead_pub_report = format!(
        "dead public item scan identified {} violation groups",
        dead_pubviolations.len()
    );
    emit_stage_detail(&dead_pub_report);
    allviolations.extend(dead_pubviolations);

    // Enforce that `use opt::` imports only appear in the strategy module.
    // All optimizer access must go through `run_outer` in strategy.rs.
    update_stage("scan direct opt crate imports");
    let opt_importviolations = scan_for_direct_opt_imports();
    let opt_import_report = format!(
        "direct opt import scan identified {} violation groups",
        opt_importviolations.len()
    );
    emit_stage_detail(&opt_import_report);
    allviolations.extend(opt_importviolations);

    // Write lint violations to a generated source file so they surface as
    // compile_error! alongside real compiler errors, rather than blocking
    // compilation entirely.  This means real type/borrow/syntax errors are
    // always visible — lint violations no longer hide them.
    let out_dir = std::env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let lint_path = Path::new(&out_dir).join("lint_errors.rs");
    if !allviolations.is_empty() {
        update_stage("report validation errors");
        let violation_count = allviolations.len();
        let mut generated = String::new();
        generated.push_str("// Auto-generated by build.rs — do not edit\n");
        for (i, violation) in allviolations.iter().enumerate() {
            // Escape the violation text for use inside a string literal.
            let escaped = violation
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            generated.push_str(&format!(
                "compile_error!(\"[lint {}/{}] {}\");\n",
                i + 1,
                violation_count,
                escaped
            ));
        }
        std::fs::write(&lint_path, generated).unwrap_or_else(|e| {
            panic!("failed to write lint errors to {}: {e}", lint_path.display());
        });
    } else {
        // No violations — write an empty file so the include! never fails.
        std::fs::write(&lint_path, "// No lint violations\n").unwrap_or_else(|e| {
            panic!("failed to write lint errors to {}: {e}", lint_path.display());
        });
    }

    update_stage("build script completed");
    emit_stage_detail("Validation checks completed without errors");
}

// This function manually checks for unused variables in the current file
fn manually_check_for_unusedvariables() {
    // Force compilation to fail with unused_variables, dead_code, and unused_imports lint
    // This ensures build.rs itself follows the strict coding policy
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let build_path = manifest_dir.join("build.rs");

    if !build_path.exists() {
        emit_stage_detail("manual lint self-check: build script source not found");
        eprintln!(
            "manual lint self-check fatal error: build script source file {:?} is missing",
            build_path
        );
        std::process::exit(1);
    }

    let deps_dir = match build_dependencies_directory() {
        Some(path) => path,
        None => {
            emit_stage_detail(
                "manual lint self-check: could not determine build dependency directory",
            );
            eprintln!(
                "manual lint self-check fatal error: unable to derive build dependency directory from OUT_DIR"
            );
            std::process::exit(1);
        }
    };

    let mut manual_lint_args = manual_lint_arguments(&build_path);
    let source_path = match manual_lint_args.pop() {
        Some(path) => path,
        None => {
            emit_stage_detail(
                "manual lint self-check: unable to obtain source path from manual lint arguments",
            );
            eprintln!(
                "manual lint self-check fatal error: manual lint argument assembly failed to include the source path"
            );
            std::process::exit(1);
        }
    };

    manual_lint_args.push(OsString::from("-L"));
    manual_lint_args.push(OsString::from(format!("dependency={}", deps_dir.display())));

    for cratename in ["grep", "walkdir"] {
        match locate_build_dependency(&deps_dir, cratename) {
            Some(artifact_path) => {
                manual_lint_args.push(OsString::from("--extern"));
                manual_lint_args.push(OsString::from(format!(
                    "{cratename}={}",
                    artifact_path.display()
                )));
            }
            None => {
                emit_stage_detail(&format!(
                    "manual lint self-check: missing rlib for dependency '{cratename}'"
                ));
                eprintln!(
                    "manual lint self-check fatal error: required dependency '{cratename}' rlib not found in {:?}",
                    deps_dir
                );
                std::process::exit(1);
            }
        }
    }

    manual_lint_args.push(source_path);
    let rustc_binary = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));

    update_stage("manual lint self-check: running rustc");
    emit_stage_detail(&format!(
        "manual lint self-check: selected rustc executable: {:?}",
        rustc_binary
    ));

    if let Some(host) = std::env::var_os("HOST") {
        emit_stage_detail(&format!(
            "manual lint self-check: HOST environment: {:?}",
            host
        ));
    }

    if let Some(target) = std::env::var_os("TARGET") {
        emit_stage_detail(&format!(
            "manual lint self-check: TARGET environment: {:?}",
            target
        ));
    }

    if let Some(triple) = std::env::var_os("CARGO_CFG_TARGET_ARCH") {
        emit_stage_detail(&format!(
            "manual lint self-check: cfg target arch: {:?}",
            triple
        ));
    }

    emit_stage_detail(&format!(
        "manual lint self-check: build context arch/os: {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    ));

    update_stage("manual lint self-check: preparing rustc command");
    emit_stage_detail(&format!(
        "manual lint self-check: command preview: {}",
        command_preview(&rustc_binary, &manual_lint_args)
    ));

    if let Ok(cwd) = std::env::current_dir() {
        emit_stage_detail(&format!(
            "manual lint self-check: current dir before spawn: {:?}",
            cwd
        ));
    }

    let mut command = std::process::Command::new(&rustc_binary);
    command.current_dir(&manifest_dir);
    command.args(&manual_lint_args);

    update_stage("manual lint self-check: invoking rustc");
    emit_stage_detail("manual lint self-check: calling Command::output() for rustc self-lint");
    let status = command.output();

    update_stage("manual lint self-check: rustc invocation returned");

    match status {
        Ok(output) => {
            emit_stage_detail(&format!(
                "manual lint self-check: rustc exit status: {:?}",
                output.status.code()
            ));
            emit_stage_detail(&format!(
                "manual lint self-check: rustc stdout bytes: {} / stderr bytes: {}",
                output.stdout.len(),
                output.stderr.len()
            ));

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("unused variable") {
                    eprintln!("\n❌ ERROR: Unused variables detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused variables are STRICTLY FORBIDDEN in this project.");
                    eprintln!(
                        "   Either use the variable or remove it completely. Underscore prefixes are NOT allowed."
                    );
                    eprintln!(
                        "   These are the only two options: fully wire it into real use, or delete it entirely."
                    );
                    eprintln!(
                        "   Do not suppress, hide, no-op, stub, or work around this lint in any other way."
                    );
                    std::process::exit(1);
                } else if stderr.contains("function is never used") {
                    eprintln!("\n❌ ERROR: Unused functions detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused functions are STRICTLY FORBIDDEN in this project.");
                    eprintln!("   Either use the function or remove it completely.");
                    eprintln!(
                        "   These are the only two options: fully wire it into real use, or delete it entirely."
                    );
                    eprintln!(
                        "   Do not suppress, hide, no-op, stub, or work around this lint in any other way."
                    );
                    std::process::exit(1);
                } else if stderr.contains("unused import") {
                    eprintln!("\n❌ ERROR: Unused imports detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused imports are STRICTLY FORBIDDEN in this project.");
                    eprintln!("   Either use the imported item or remove the import completely.");
                    eprintln!(
                        "   These are the only two options: fully wire it into real use, or delete it entirely."
                    );
                    eprintln!(
                        "   Do not suppress, hide, no-op, stub, or work around this lint in any other way."
                    );
                    std::process::exit(1);
                } else {
                    eprintln!(
                        "manual lint self-check fatal error: rustc self-lint exited with status {}",
                        output
                            .status
                            .code()
                            .map(|code| code.to_string())
                            .unwrap_or_else(|| String::from("<signal>"))
                    );
                    if !output.stderr.is_empty() {
                        eprintln!("rustc self-lint stderr:\n{}", stderr);
                    }
                    std::process::exit(1);
                }
            } else {
                emit_stage_detail("Completed rustc self-lint for build.rs");
            }
        }
        Err(err) => {
            emit_stage_detail(&format!(
                "manual lint self-check: failed to start rustc self-lint command: {err}"
            ));
            eprintln!(
                "manual lint self-check fatal error: failed to spawn rustc self-lint command: {err}"
            );
            std::process::exit(1);
        }
    }
}

fn manual_lint_arguments(build_path: &Path) -> Vec<OsString> {
    vec![
        OsString::from("--edition"),
        OsString::from("2024"),
        // Lint without linking to avoid toolchain linker/bitcode mismatches.
        OsString::from("--emit"),
        OsString::from("metadata"),
        OsString::from("-D"),
        OsString::from("unused_variables"),
        OsString::from("-D"),
        OsString::from("dead_code"),
        OsString::from("-D"),
        OsString::from("unused_imports"),
        OsString::from("--crate-type"),
        OsString::from("bin"),
        OsString::from("--error-format"),
        OsString::from("human"),
        build_path.as_os_str().to_os_string(),
    ]
}

fn build_dependencies_directory() -> Option<PathBuf> {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR")?);
    let profile_dir = out_dir.ancestors().nth(3)?;
    Some(profile_dir.join("deps"))
}

#[allow(clippy::collapsible_if)]
fn locate_build_dependency(deps_dir: &Path, cratename: &str) -> Option<PathBuf> {
    let prefix = format!("lib{cratename}-");
    let mut candidate: Option<PathBuf> = None;
    let mut candidate_mtime = std::time::SystemTime::UNIX_EPOCH;

    if let Ok(entries) = std::fs::read_dir(deps_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rlib") {
                continue;
            }

            let filename = match path.file_name().and_then(|name| name.to_str()) {
                Some(name) => name,
                None => continue,
            };

            if filename.starts_with(&prefix) {
                if let Ok(metadata) = std::fs::metadata(&path) {
                    if let Ok(mtime) = metadata.modified() {
                        if candidate.is_none() || mtime > candidate_mtime {
                            candidate = Some(path);
                            candidate_mtime = mtime;
                        }
                    }
                }
            }
        }
    }

    candidate
}

fn command_preview(program: &OsStr, args: &[OsString]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(format!("{program:?}"));
    for arg in args {
        parts.push(format!("{arg:?}"));
    }
    parts.join(" ")
}

fn scan_for_underscore_prefixes() -> Vec<String> {
    // Regex pattern to find underscore prefixed variable names.
    // This pattern needs to be more generalized to catch all underscore-prefixed variables,
    // especially in match statements and destructuring patterns
    // Allow SIMD intrinsics like _mm512_* while still flagging underscore-prefixed bindings.
    // (grep regex doesn't support look-around, so filtering happens in the collector.)
    let pattern = r"\b_[a-zA-Z0-9_]+\b";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            // Use `walkdir` to find all Rust files, replacing the `find` command.
            // This is more portable and robust.
            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok()) // Ignore any errors during directory traversal.
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path())) // Exclude ignored directories.
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            // Keep only .rs files.
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = ViolationCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                // Process results
                if let Some(error_message) = collector.check_and_get_error_message() {
                    // Add this error to our collection instead of returning immediately
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            // If there's an error creating the matcher, report it but don't return early
            allviolations.push(format!(
                "Error creating regex matcher for underscore prefixes: {}",
                e
            ));
        }
    }

    // Return all violations found
    allviolations
}

fn scan_for_disallowed_let_patterns() -> Vec<String> {
    let pattern = r"\blet\s+(?:mut\s+)?_\s*=";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DisallowedLetCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating regex matcher for disallowed let patterns: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_noop_touch_patterns() -> Vec<String> {
    let pattern = r"^\s*[A-Za-z_][A-Za-z0-9_:<>]*\s*;\s*$|^\s*[^=]*\.\s*(as_ref|as_mut|as_ptr|clone|to_string|to_owned|as_bytes|as_str|len|is_empty)\s*\(\s*\)\s*;\s*$";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = NoopTouchCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                // Post-filter: remove false positives from multi-line let bindings
                // where cargo fmt splits "let x =\n    expr;" across two lines.
                if !collector.violations.is_empty() {
                    let stripped_str = String::from_utf8_lossy(&stripped);
                    let lines: Vec<&str> = stripped_str.split('\n').collect();
                    collector.violations.retain(|v| {
                        if let Some(line_no) =
                            v.split(':').next().and_then(|s| s.parse::<usize>().ok())
                        {
                            if line_no >= 2 {
                                let prev = lines
                                    .get(line_no - 2)
                                    .map(|l: &&str| l.trim())
                                    .unwrap_or("");
                                if prev.ends_with('=') {
                                    return false;
                                }
                            }
                        }
                        true
                    });
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating regex matcher for no-op touch patterns: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_tuplewildcard_patterns() -> Vec<String> {
    let pattern = r"\blet\s*\([^)]*\b_\b[^)]*\)\s*(?::[^=]*)?=";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = TupleWildcardCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating regex matcher for tuple wildcard patterns: {}",
                e
            ));
        }
    }

    allviolations
}

fn is_doc_comment(line: &str) -> bool {
    line.trim_start().starts_with("///")
}

fn scan_for_forbidden_comment_patterns() -> Vec<String> {
    // Strip string literals first (via strip_strings_only) so that patterns
    // like `"// DEPRECATED"` inside strings never produce false positives.
    let mut allviolations = Vec::new();

    let forbiddenwords_pattern = r"(//|/\*|///).*(?:DEPRECATED|CRITICAL|FIXED|CORRECTED|FIX|FIXES|NEW|CHANGED|CHANGES|CHANGE|MODIFIED|MODIFIES|MODIFY|UPDATED|UPDATES|UPDATE)";
    // 2. Pattern to catch ** in comments (excluding doc comments)
    let stars_pattern = r"(//|/\*).*\*\*";
    // 3. Pattern to catch comments for uppercase ratio enforcement
    let all_caps_pattern = r"(//|/\*|///).*";
    let forbidden_matcher = RegexMatcher::new_line_matcher(forbiddenwords_pattern);
    let stars_matcher = RegexMatcher::new_line_matcher(stars_pattern);
    let all_caps_matcher = RegexMatcher::new_line_matcher(all_caps_pattern);

    // Single file walk — strip once per file, run all 3 sub-scans.
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
        .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
        .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
        .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // Strip only strings — comments are preserved for pattern matching.
        let stripped = strip_strings_only(&content);

        let mut searcher = Searcher::new();

        if let Ok(ref m) = forbidden_matcher {
            let mut collector = ForbiddenCommentCollector::new(path, true);
            if searcher
                .search_reader(m, Cursor::new(&stripped), &mut collector)
                .is_ok()
            {
                if let Some(msg) = collector.check_and_get_error_message() {
                    allviolations.push(msg);
                }
            }
        }

        if let Ok(ref m) = stars_matcher {
            let mut collector = ForbiddenCommentCollector::new(path, false);
            if searcher
                .search_reader(m, Cursor::new(&stripped), &mut collector)
                .is_ok()
            {
                if let Some(msg) = collector.check_and_get_error_message() {
                    allviolations.push(msg);
                }
            }
        }

        if let Ok(ref m) = all_caps_matcher {
            let mut collector = CustomUppercaseCollector::new(path);
            if searcher
                .search_reader(m, Cursor::new(&stripped), &mut collector)
                .is_ok()
            {
                if let Some(msg) = collector.check_and_get_error_message() {
                    allviolations.push(msg);
                }
            }
        }
    }

    allviolations
}

fn scan_for_allow_dead_code() -> Vec<String> {
    // Regex pattern to find forbidden allow attributes.
    // Matches:
    // - #[allow(dead_code)] or #![allow(dead_code)]
    // - #[allow(unused)]
    // - #[allow(unused_imports)]
    // - #[allow(unused_variables)]
    // - #[allow(clippy::*)] (any clippy lint suppression)
    // - #[allow(..., unused, ...)] (inside lists)
    let pattern = r"#!?\s*\[\s*allow\s*\([^)]*\b(dead_code|unused|unused_imports|unused_variables|clippy::\w+)\b[^)]*\)\s*\]";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path())) // Exclude ignored directories
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DeadCodeCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    // Handle search errors gracefully
                    continue;
                }

                // Process results
                if let Some(error_message) = collector.check_and_get_error_message() {
                    // Add this error to our collection instead of returning immediately
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            // If there's an error creating the matcher, report it but don't return early
            allviolations.push(format!("Error creating dead code regex matcher: {}", e));
        }
    }

    // Return all violations found
    allviolations
}

fn scan_for_ignored_tests() -> Vec<String> {
    // Regex pattern to find #[ignore] test attributes in all forms:
    // - #[ignore]
    // - #[ignore = "reason"]
    // - #[ignore("reason")]
    // - #[cfg_attr(..., ignore)]
    // - #[cfg_attr(..., ignore = "reason")]
    let pattern = r"#\s*\[.*\bignore\b";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path())) // Exclude ignored directories
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = IgnoredTestCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    // Handle search errors gracefully
                    continue;
                }

                // Process results
                if let Some(error_message) = collector.check_and_get_error_message() {
                    // Add this error to our collection instead of returning immediately
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            // If there's an error creating the matcher, report it but don't return early
            allviolations.push(format!("Error creating ignored tests regex matcher: {}", e));
        }
    }

    // Return all violations found
    allviolations
}

fn scan_for_todo_comments() -> Vec<String> {
    // Regex pattern to find deferred-work comment markers (case insensitive)
    // Matches standard deferred-work markers in comments
    let pattern = r"(?i)//.*\b(TODO|FIXME|XXX|for now)\b";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_strings_only(&content);

                let mut collector = TodoCommentCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!("Error creating TODO comment regex matcher: {}", e));
        }
    }

    allviolations
}

fn scan_for_drop_in_build_scripts() -> Vec<String> {
    let pattern = r"\bdrop\s*\(";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| {
                    e.path()
                        .file_name()
                        .is_some_and(|name| name == OsStr::new("build.rs"))
                })
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DropUsageCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating drop usage regex matcher for build scripts: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_drop_usage() -> Vec<String> {
    let pattern = r"\bdrop\s*\(";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DropUsageCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!("Error creating drop usage regex matcher: {}", e));
        }
    }

    allviolations
}

fn scan_for_empty_control_blocks() -> Vec<String> {
    let mut allviolations = Vec::new();

    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
        .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
        .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let source = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(_) => continue,
        };

        let mut collector = EmptyBlockCollector::new(path);
        let violations = find_empty_control_blocks(&source);
        collector.violations.extend(violations);

        if let Some(error_message) = collector.check_and_get_error_message() {
            allviolations.push(error_message);
        }
    }

    allviolations
}

fn scan_for_debug_assert_usage() -> Vec<String> {
    let pattern = r"\bdebug_assert!\s*\(";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DebugAssertCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!("Error creating debug_assert regex matcher: {}", e));
        }
    }

    allviolations
}

fn scan_for_meaningless_conditionals() -> Vec<String> {
    // Match single-line if-else expressions to find candidates
    // The Sink implementation will then compare the branches
    let pattern = r"if\s+.+\s*\{[^}]+\}\s*else\s*\{[^}]+\}";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = MeaninglessConditionalCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating meaningless conditional regex matcher: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_dead_code_anchors() -> Vec<String> {
    // Catches `const _: fn() = some_func_name;` — the pattern used to keep
    // otherwise-dead preserve_* functions alive at compile time.
    let pattern = r"const\s+_\s*:\s*fn\s*\(\s*\)\s*=\s*\w+\s*;";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DeadCodeAnchorCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating dead-code anchor regex matcher: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_no_effect_allow() -> Vec<String> {
    let pattern = r"#\s*\[\s*allow\s*\(\s*clippy\s*::\s*no_effect\s*\)\s*\]";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = NoEffectCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating no_effect allow regex matcher: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_omitted_for_brevity() -> Vec<String> {
    // Match various forms of "omitted for brevity" and similar placeholders
    // Pattern includes: "omitted for brevity", "not shown", "elided", "truncated", "abbreviated", etc.
    let pattern = r"(omitted\s+for\s+brevity|not\s+shown|elided\s+for|truncated\s+for|abbreviated\s+for|removed\s+for\s+brevity|excluded\s+for\s+brevity|skipped\s+for\s+brevity|hidden\s+for\s+brevity)";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_strings_only(&content);

                let mut collector = OmittedForBrevityCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating omitted for brevity regex matcher: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_deprecated() -> Vec<String> {
    let pattern = r"#\s*\[\s*deprecated";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = DeprecatedCollector::new(path);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating deprecated attribute regex matcher: {}",
                e
            ));
        }
    }

    allviolations
}

fn scan_for_placeholder_stubs() -> Vec<String> {
    // Match functions returning Ok(Self { with empty collections
    // Pattern: Ok(Self { followed by lines with vec![], HashMap::new(), etc., and closing })
    let pattern = r"Ok\(Self\s*\{";
    let mut allviolations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let stripped = strip_comments_and_strings_for_content(&content);

                let mut collector = PlaceholderStubCollector::new(path, &content);

                if searcher
                    .search_reader(&matcher, Cursor::new(&stripped), &mut collector)
                    .is_err()
                {
                    continue;
                }

                // Check if the collector found any violations
                if let Some(error_message) = collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!(
                "Error creating placeholder stub regex matcher: {}",
                e
            ));
        }
    }

    allviolations
}

fn find_empty_control_blocks(source: &str) -> Vec<String> {
    let tokens_sanitized = strip_comments_and_strings_for_tokens(source);
    let content_sanitized = strip_comments_and_strings_for_content(source);
    let tokens = tokenize_for_empty_block_scan(&tokens_sanitized);
    let lines: Vec<&str> = source.lines().collect();
    let mut brace_stack: Vec<usize> = Vec::new();
    let mut matches = vec![None; tokens.len()];

    for (idx, token) in tokens.iter().enumerate() {
        match token.kind {
            EmptyBlockTokenKind::OpenBrace => brace_stack.push(idx),
            EmptyBlockTokenKind::CloseBrace => {
                if let Some(open_idx) = brace_stack.pop() {
                    matches[open_idx] = Some(idx);
                }
            }
            _ => {}
        }
    }

    let mut violations = Vec::new();
    for (idx, token) in tokens.iter().enumerate() {
        let Some(close_idx) = matches.get(idx).and_then(|m| *m) else {
            continue;
        };

        if !matches!(token.kind, EmptyBlockTokenKind::OpenBrace) {
            continue;
        }

        let open_offset = token.offset + 1;
        let close_offset = tokens[close_idx].offset;
        if close_offset <= open_offset {
            continue;
        }

        if !content_sanitized[open_offset..close_offset]
            .iter()
            .all(|byte| byte.is_ascii_whitespace())
        {
            continue;
        }

        let depth = token.depth;
        let mut control_keyword = None;
        for prev_idx in (0..idx).rev() {
            let prev_token = &tokens[prev_idx];
            if prev_token.depth < depth {
                break;
            }
            if prev_token.depth > depth {
                continue;
            }
            match prev_token.kind {
                EmptyBlockTokenKind::Semicolon => break,
                EmptyBlockTokenKind::OpenBrace | EmptyBlockTokenKind::CloseBrace => break,
                EmptyBlockTokenKind::Arrow => break,
                EmptyBlockTokenKind::Ident(ref ident) => {
                    if matches!(
                        ident.as_str(),
                        "if" | "else" | "for" | "while" | "loop" | "match"
                    ) {
                        control_keyword = Some(ident.as_str());
                        break;
                    }
                }
            }
        }

        if control_keyword.is_none() {
            continue;
        }

        let line_number = token.line;
        let line_text = lines
            .get(line_number.saturating_sub(1))
            .copied()
            .unwrap_or("")
            .trim_end();
        violations.push(format!("{line_number}:{line_text}"));
    }

    violations
}

fn strip_comments_and_strings_for_tokens(source: &str) -> Vec<u8> {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        LineComment,
        BlockComment(usize),
        StringLiteral,
        CharLiteral,
        RawString(usize),
    }

    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let mut state = State::Normal;

    while i < bytes.len() {
        let b = bytes[i];
        match state {
            State::Normal => {
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::LineComment;
                    continue;
                }
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::BlockComment(1);
                    continue;
                }
                if let Some((hashes, consumed)) = raw_string_start(bytes, i) {
                    out.extend(std::iter::repeat_n(b' ', consumed));
                    i += consumed;
                    state = State::RawString(hashes);
                    continue;
                }
                if b == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::StringLiteral;
                    continue;
                }
                if b == b'"' {
                    out.push(b' ');
                    i += 1;
                    state = State::StringLiteral;
                    continue;
                }
                if b == b'\'' {
                    out.push(b' ');
                    i += 1;
                    state = State::CharLiteral;
                    continue;
                }
                out.push(b);
                i += 1;
            }
            State::LineComment => {
                if b == b'\n' {
                    out.push(b'\n');
                    i += 1;
                    state = State::Normal;
                } else {
                    out.push(b' ');
                    i += 1;
                }
            }
            State::BlockComment(depth) => {
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::BlockComment(depth + 1);
                    continue;
                }
                if b == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    if depth == 1 {
                        state = State::Normal;
                    } else {
                        state = State::BlockComment(depth - 1);
                    }
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            State::StringLiteral => {
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    continue;
                }
                if b == b'"' {
                    out.push(b' ');
                    i += 1;
                    state = State::Normal;
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            State::CharLiteral => {
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    continue;
                }
                if b == b'\'' {
                    out.push(b' ');
                    i += 1;
                    state = State::Normal;
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            State::RawString(hashes) => {
                if b == b'"' && raw_string_end(bytes, i, hashes) {
                    out.push(b' ');
                    i += 1;
                    out.extend(std::iter::repeat_n(b' ', hashes));
                    i += hashes;
                    state = State::Normal;
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
        }
    }

    out
}

fn strip_comments_and_strings_for_content(source: &str) -> Vec<u8> {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        LineComment,
        BlockComment(usize),
        StringLiteral,
        CharLiteral,
        RawString(usize),
    }

    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let mut state = State::Normal;

    while i < bytes.len() {
        let b = bytes[i];
        match state {
            State::Normal => {
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::LineComment;
                    continue;
                }
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::BlockComment(1);
                    continue;
                }
                if let Some((hashes, consumed)) = raw_string_start(bytes, i) {
                    out.extend(std::iter::repeat_n(b'x', consumed));
                    i += consumed;
                    state = State::RawString(hashes);
                    continue;
                }
                if b == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    out.push(b'x');
                    out.push(b'x');
                    i += 2;
                    state = State::StringLiteral;
                    continue;
                }
                if b == b'"' {
                    out.push(b'x');
                    i += 1;
                    state = State::StringLiteral;
                    continue;
                }
                if b == b'\'' {
                    // Distinguish char literals ('x', '\n') from lifetime ticks ('a).
                    let is_char_lit = if i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                        // Escape sequence: '\n', '\x41', '\u{...}' — closing tick within 10 bytes
                        bytes[i + 2..std::cmp::min(i + 12, bytes.len())]
                            .iter()
                            .any(|&c| c == b'\'')
                    } else if i + 2 < bytes.len() && bytes[i + 2] == b'\'' {
                        // Simple char: 'x'
                        true
                    } else {
                        false
                    };
                    if is_char_lit {
                        out.push(b'x');
                        i += 1;
                        state = State::CharLiteral;
                        continue;
                    }
                    // Lifetime tick — emit as-is.
                    out.push(b);
                    i += 1;
                    continue;
                }
                out.push(b);
                i += 1;
            }
            State::LineComment => {
                if b == b'\n' {
                    out.push(b'\n');
                    i += 1;
                    state = State::Normal;
                } else {
                    out.push(b' ');
                    i += 1;
                }
            }
            State::BlockComment(depth) => {
                if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    state = State::BlockComment(depth + 1);
                    continue;
                }
                if b == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    out.push(b' ');
                    out.push(b' ');
                    i += 2;
                    if depth == 1 {
                        state = State::Normal;
                    } else {
                        state = State::BlockComment(depth - 1);
                    }
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b' ');
                }
                i += 1;
            }
            State::StringLiteral => {
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b'x');
                    out.push(b'x');
                    i += 2;
                    continue;
                }
                if b == b'"' {
                    out.push(b'x');
                    i += 1;
                    state = State::Normal;
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b'x');
                }
                i += 1;
            }
            State::CharLiteral => {
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b'x');
                    out.push(b'x');
                    i += 2;
                    continue;
                }
                if b == b'\'' {
                    out.push(b'x');
                    i += 1;
                    state = State::Normal;
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b'x');
                }
                i += 1;
            }
            State::RawString(hashes) => {
                if b == b'"' && raw_string_end(bytes, i, hashes) {
                    out.push(b'x');
                    i += 1;
                    out.extend(std::iter::repeat_n(b'x', hashes));
                    i += hashes;
                    state = State::Normal;
                    continue;
                }
                if b == b'\n' {
                    out.push(b'\n');
                } else {
                    out.push(b'x');
                }
                i += 1;
            }
        }
    }

    out
}

#[allow(dead_code)]
fn strip_strings_only(source: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        StringLiteral,
        CharLiteral,
        RawString(usize),
    }

    let bytes = source.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let mut state = State::Normal;

    while i < bytes.len() {
        let b = bytes[i];
        match state {
            State::Normal => {
                if let Some((hashes, consumed)) = raw_string_start(bytes, i) {
                    out.extend(std::iter::repeat_n(b'x', consumed));
                    i += consumed;
                    state = State::RawString(hashes);
                    continue;
                }
                if b == b'b' && i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    out.push(b'x');
                    out.push(b'x');
                    i += 2;
                    state = State::StringLiteral;
                    continue;
                }
                if b == b'"' {
                    out.push(b'x');
                    i += 1;
                    state = State::StringLiteral;
                    continue;
                }
                if b == b'\'' {
                    let is_char_lit = if i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                        bytes[i + 2..std::cmp::min(i + 12, bytes.len())]
                            .iter()
                            .any(|&c| c == b'\'')
                    } else {
                        i + 2 < bytes.len() && bytes[i + 2] == b'\''
                    };
                    if is_char_lit {
                        out.push(b'x');
                        i += 1;
                        state = State::CharLiteral;
                        continue;
                    }
                }
                out.push(b);
                i += 1;
            }
            State::StringLiteral => {
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b'x');
                    out.push(b'x');
                    i += 2;
                    continue;
                }
                if b == b'"' {
                    out.push(b'x');
                    i += 1;
                    state = State::Normal;
                    continue;
                }
                out.push(if b == b'\n' { b'\n' } else { b'x' });
                i += 1;
            }
            State::CharLiteral => {
                if b == b'\\' && i + 1 < bytes.len() {
                    out.push(b'x');
                    out.push(b'x');
                    i += 2;
                    continue;
                }
                if b == b'\'' {
                    out.push(b'x');
                    i += 1;
                    state = State::Normal;
                    continue;
                }
                out.push(if b == b'\n' { b'\n' } else { b'x' });
                i += 1;
            }
            State::RawString(hashes) => {
                if b == b'"' && raw_string_end(bytes, i, hashes) {
                    out.push(b'x');
                    i += 1;
                    out.extend(std::iter::repeat_n(b'x', hashes));
                    i += hashes;
                    state = State::Normal;
                    continue;
                }
                out.push(if b == b'\n' { b'\n' } else { b'x' });
                i += 1;
            }
        }
    }

    String::from_utf8(out).expect("sanitized Rust source should remain UTF-8")
}

fn raw_string_start(bytes: &[u8], idx: usize) -> Option<(usize, usize)> {
    let offset = if bytes.get(idx) == Some(&b'r') {
        1
    } else if bytes.get(idx) == Some(&b'b') && bytes.get(idx + 1) == Some(&b'r') {
        2
    } else {
        return None;
    };

    let mut hashes = 0usize;
    let mut j = idx + offset;
    while bytes.get(j) == Some(&b'#') {
        hashes += 1;
        j += 1;
    }

    if bytes.get(j) != Some(&b'"') {
        return None;
    }

    Some((hashes, j + 1 - idx))
}

fn raw_string_end(bytes: &[u8], idx: usize, hashes: usize) -> bool {
    if bytes.get(idx) != Some(&b'"') {
        return false;
    }
    for h in 0..hashes {
        if bytes.get(idx + 1 + h) != Some(&b'#') {
            return false;
        }
    }
    true
}

fn tokenize_for_empty_block_scan(sanitized: &[u8]) -> Vec<EmptyBlockToken> {
    let mut tokens = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;

    while i < sanitized.len() {
        let b = sanitized[i];
        if b == b'\n' {
            line += 1;
            i += 1;
            continue;
        }

        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < sanitized.len()
                && (sanitized[i].is_ascii_alphanumeric() || sanitized[i] == b'_')
            {
                i += 1;
            }
            let ident = String::from_utf8_lossy(&sanitized[start..i]).to_string();
            tokens.push(EmptyBlockToken {
                kind: EmptyBlockTokenKind::Ident(ident),
                line,
                offset: start,
                depth: 0,
            });
            continue;
        }

        if b == b'=' && sanitized.get(i + 1) == Some(&b'>') {
            tokens.push(EmptyBlockToken {
                kind: EmptyBlockTokenKind::Arrow,
                line,
                offset: i,
                depth: 0,
            });
            i += 2;
            continue;
        }

        let kind = match b {
            b'{' => Some(EmptyBlockTokenKind::OpenBrace),
            b'}' => Some(EmptyBlockTokenKind::CloseBrace),
            b';' => Some(EmptyBlockTokenKind::Semicolon),
            _ => None,
        };

        if let Some(kind) = kind {
            tokens.push(EmptyBlockToken {
                kind,
                line,
                offset: i,
                depth: 0,
            });
        }
        i += 1;
    }

    let mut depth = 0usize;
    for token in &mut tokens {
        token.depth = depth;
        match token.kind {
            EmptyBlockTokenKind::OpenBrace => depth += 1,
            EmptyBlockTokenKind::CloseBrace => depth = depth.saturating_sub(1),
            _ => {}
        }
    }

    tokens
}

fn is_in_hidden_directory(path: impl AsRef<Path>) -> bool {
    path.as_ref().components().any(|component| {
        if let Component::Normal(name) = component {
            let name_str = name.to_string_lossy();
            name_str.starts_with('.')
        } else {
            false
        }
    })
}

fn is_in_target_directory(path: impl AsRef<Path>) -> bool {
    path.as_ref().components().any(|component| {
        matches!(
            component,
            Component::Normal(name)
                if name
                    .to_str()
                    .is_some_and(|segment| segment == "target" || segment.starts_with("target-"))
        )
    })
}

fn is_in_ignored_directory(path: impl AsRef<Path>) -> bool {
    is_in_target_directory(path.as_ref()) || is_in_hidden_directory(path.as_ref())
}

#[derive(Clone)]
struct SimpleBoolHelper {
    params: Vec<String>,
    body_expr: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SimpleMacroAliasKind {
    IdentityExpr,
    BlackBoxTouch,
    SizeOfValTouch,
}

#[derive(Clone)]
struct SimpleMacroAlias {
    line_number: usize,
    kind: SimpleMacroAliasKind,
}

struct UnsignedSymbolHints {
    value_names: HashSet<String>,
    method_names: HashSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AddedCfgTestSite {
    file: String,
    line: usize,
    item_preview: String,
    source_label: &'static str,
}

fn scan_for_inert_touch_calls() -> Vec<String> {
    let mut allviolations = Vec::new();

    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !is_in_ignored_directory(e.path()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let sanitized_bytes = strip_comments_and_strings_for_content(&content);
        let sanitized = String::from_utf8_lossy(&sanitized_bytes);
        let mut file_violations = Vec::new();

        for (line_idx, line) in sanitized.lines().enumerate() {
            let trimmed = line.trim();
            let Some(statement) = trimmed.strip_suffix(';').map(str::trim) else {
                continue;
            };
            if statement.contains('=') {
                continue;
            }
            let Some((callee, args)) = extract_call_expression(statement) else {
                continue;
            };
            let is_black_box = matches!(
                callee.as_str(),
                "std::hint::black_box" | "core::hint::black_box"
            );
            let is_size_of_val = matches!(
                callee.as_str(),
                "std::mem::size_of_val" | "core::mem::size_of_val"
            );
            if !is_black_box && !is_size_of_val {
                continue;
            }
            if !is_simple_inert_touch_argument(&args) {
                continue;
            }

            let reason = if is_black_box {
                "standalone black_box touch on a simple path/reference is a fake use"
            } else {
                "standalone size_of_val touch on a simple path/reference is a fake use"
            };
            file_violations.push(format!("{}:{} -> {}", line_idx + 1, trimmed, reason));
        }

        if !file_violations.is_empty() {
            allviolations.push(format_grouped_scan_error(
                "inert touch calls",
                path,
                &file_violations,
                "Use the value meaningfully or delete it. Standalone black_box/size_of_val touches do not count as real use.\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.",
            ));
        }
    }

    allviolations
}

fn scan_for_cfg_test_evasion() -> Vec<String> {
    let mut allviolations = Vec::new();
    let mut sites: HashSet<AddedCfgTestSite> = HashSet::new();

    for site in
        collect_added_cfg_test_sites(&["diff", "--name-only", "HEAD"], &["HEAD"], "working tree")
    {
        sites.insert(site);
    }
    for site in collect_added_cfg_test_sites(
        &["diff", "--name-only", "@{upstream}..HEAD"],
        &["@{upstream}..HEAD"],
        "unpushed commit",
    ) {
        sites.insert(site);
    }

    if sites.is_empty() {
        return allviolations;
    }

    let mut grouped: HashMap<String, Vec<AddedCfgTestSite>> = HashMap::new();
    for site in sites {
        grouped.entry(site.file.clone()).or_default().push(site);
    }

    for (file, mut file_sites) in grouped {
        file_sites.sort_by_key(|site| site.line);
        let path = Path::new(&file);
        let violations = file_sites
            .into_iter()
            .map(|site| {
                format!(
                    "{}:{} -> newly-added #[cfg(test)] gates `{}`",
                    site.line, site.source_label, site.item_preview
                )
            })
            .collect::<Vec<_>>();
        allviolations.push(format_grouped_scan_error(
            "newly-added #[cfg(test)] production gates",
            path,
            &violations,
            "Do not mark production imports, fields, constants, constructors, or helper methods as #[cfg(test)] to suppress dead_code in normal builds. Either wire the code into production, delete it, or keep the test-only logic inside a real #[cfg(test)] test module.\n   These are the only two options: fully wire it into real use, or delete it entirely.\n   Do not suppress, hide, no-op, stub, or work around this lint in any other way.",
        ));
    }

    allviolations
}

fn collect_added_cfg_test_sites(
    changed_files_args: &[&str],
    patch_spec: &[&str],
    source_label: &'static str,
) -> Vec<AddedCfgTestSite> {
    let changed_files = git_changed_files_set(changed_files_args);
    let mut sites = Vec::new();

    for file in changed_files {
        if !file.starts_with("src/") || !file.ends_with(".rs") {
            continue;
        }

        let mut args = vec!["diff", "--unified=3"];
        args.extend_from_slice(patch_spec);
        args.push("--");
        args.push(file.as_str());

        let output = match std::process::Command::new("git").args(&args).output() {
            Ok(output) => output,
            Err(_) => continue,
        };
        if !output.status.success() {
            continue;
        }

        let patch = String::from_utf8_lossy(&output.stdout);
        sites.extend(parse_added_cfg_test_sites(&file, &patch, source_label));
    }

    sites
}

fn git_changed_files_set(args: &[&str]) -> HashSet<String> {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_added_cfg_test_sites(
    file: &str,
    patch: &str,
    source_label: &'static str,
) -> Vec<AddedCfgTestSite> {
    let mut sites = Vec::new();
    let lines: Vec<&str> = patch.lines().collect();
    let mut new_line_num = 0usize;
    let mut in_hunk = false;
    let mut idx = 0usize;

    while idx < lines.len() {
        let line = lines[idx];
        if let Some(start) = parse_unified_hunk_new_start(line) {
            new_line_num = start;
            in_hunk = true;
            idx += 1;
            continue;
        }

        if !in_hunk {
            idx += 1;
            continue;
        }

        if let Some(content) = line.strip_prefix('+') {
            let trimmed = content.trim();
            if is_cfg_test_attr_line(trimmed) {
                let preview = find_cfg_test_target_preview(&lines, idx + 1);
                if let Some(item_preview) = preview {
                    if !is_allowed_cfg_test_target_line(&item_preview) {
                        sites.push(AddedCfgTestSite {
                            file: file.to_string(),
                            line: new_line_num,
                            item_preview,
                            source_label,
                        });
                    }
                }
            }
            new_line_num += 1;
            idx += 1;
            continue;
        }

        if line.starts_with(' ') {
            new_line_num += 1;
            idx += 1;
            continue;
        }

        if line.starts_with('-') || line.starts_with('\\') {
            idx += 1;
            continue;
        }

        idx += 1;
    }

    sites
}

fn parse_unified_hunk_new_start(line: &str) -> Option<usize> {
    if !line.starts_with("@@") {
        return None;
    }
    let plus_idx = line.find('+')?;
    let rest = &line[plus_idx + 1..];
    let end_idx = rest
        .find(|ch: char| ch == ',' || ch.is_ascii_whitespace())
        .unwrap_or(rest.len());
    rest[..end_idx].parse::<usize>().ok()
}

fn is_cfg_test_attr_line(line: &str) -> bool {
    normalize_no_whitespace(line) == "#[cfg(test)]"
}

fn find_cfg_test_target_preview(lines: &[&str], mut idx: usize) -> Option<String> {
    while idx < lines.len() {
        let line = lines[idx];
        if line.starts_with("@@") || line.starts_with("diff --git") {
            return None;
        }
        let Some(content) = line.strip_prefix('+').or_else(|| line.strip_prefix(' ')) else {
            idx += 1;
            continue;
        };
        let trimmed = content.trim();
        if trimmed.is_empty() || trimmed.starts_with("#[") {
            idx += 1;
            continue;
        }
        return Some(trimmed.to_string());
    }
    None
}

fn is_allowed_cfg_test_target_line(line: &str) -> bool {
    let normalized = collapse_whitespace(line);
    if normalized.starts_with("mod ") {
        return normalized.contains("test");
    }
    if normalized.starts_with("pub mod ") || normalized.starts_with("pub(crate) mod ") {
        return normalized.contains("test");
    }
    false
}

fn scan_for_suspicious_macro_wrappers() -> Vec<String> {
    let mut allviolations = Vec::new();

    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !is_in_ignored_directory(e.path()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let sanitized_bytes = strip_comments_and_strings_for_content(&content);
        let sanitized = String::from_utf8_lossy(&sanitized_bytes);
        let macros = collect_simple_macro_aliases(&sanitized);
        let mut file_violations = Vec::new();

        for (name, alias) in macros {
            let reason = match alias.kind {
                SimpleMacroAliasKind::IdentityExpr => {
                    "macro_rules! expression passthrough wrapper hides the real assertion/use"
                }
                SimpleMacroAliasKind::BlackBoxTouch => {
                    "macro_rules! wrapper performs only a black_box touch"
                }
                SimpleMacroAliasKind::SizeOfValTouch => {
                    "macro_rules! wrapper performs only a size_of_val touch"
                }
            };
            file_violations.push(format!(
                "{}:macro `{}` -> {}",
                alias.line_number, name, reason
            ));
        }

        if !file_violations.is_empty() {
            allviolations.push(format_grouped_scan_error(
                "suspicious macro wrappers",
                path,
                &file_violations,
                "Do not hide assertions or fake-use touches behind local passthrough macros.",
            ));
        }
    }

    allviolations
}

fn scan_for_assertion_alias_evasions() -> Vec<String> {
    let mut allviolations = Vec::new();

    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !is_in_ignored_directory(e.path()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let sanitized_bytes = strip_comments_and_strings_for_content(&content);
        let sanitized = String::from_utf8_lossy(&sanitized_bytes).to_string();
        let helpers = collect_simple_bool_helpers(&sanitized);
        let macros = collect_simple_macro_aliases(&sanitized);
        let line_depths = compute_line_brace_depths(&content);
        let unsigned_hints = collect_unsigned_symbol_hints(&sanitized);
        let mut file_violations = Vec::new();

        for target in collect_boolean_macro_targets(&sanitized) {
            let local_aliases =
                collect_prior_bool_aliases(&sanitized, &line_depths, target.line_number);
            let expanded = expand_assertion_expression(
                &target.expression,
                &local_aliases,
                &helpers,
                &macros,
                0,
            );
            let Some(reason) =
                detect_resolved_assertion_violation(&target.expression, &expanded, &unsigned_hints)
            else {
                continue;
            };
            let original = truncate_for_display(&collapse_whitespace(&target.expression), 160);
            let resolved = truncate_for_display(&collapse_whitespace(&expanded), 160);
            file_violations.push(format!(
                "{}:{}({}) -> {} -> {}",
                target.line_number, target.macro_name, original, resolved, reason
            ));
        }

        if !file_violations.is_empty() {
            allviolations.push(format_grouped_scan_error(
                "assertion alias evasions",
                path,
                &file_violations,
                "Do not route tautological or fake-use assertions through locals, helpers, or passthrough macros.",
            ));
        }
    }

    allviolations
}

fn format_grouped_scan_error(
    title: &str,
    path: &Path,
    violations: &[String],
    guidance: &str,
) -> String {
    let filename = path.to_str().unwrap_or("?");
    let mut error_msg = format!(
        "\n❌ ERROR: Found {} {} in {}:\n",
        violations.len(),
        title,
        filename
    );
    for violation in violations {
        error_msg.push_str(&format!("   {violation}\n"));
    }
    error_msg.push_str("\n⚠️ ");
    error_msg.push_str(guidance);
    error_msg.push('\n');
    error_msg
}

fn collect_simple_macro_aliases(source: &str) -> HashMap<String, SimpleMacroAlias> {
    let mut aliases = HashMap::new();
    let bytes = source.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        let Some(rel) = source[idx..].find("macro_rules!") else {
            break;
        };
        let macro_idx = idx + rel;
        if macro_idx > 0 && is_ident_byte(bytes[macro_idx - 1]) {
            idx = macro_idx + 1;
            continue;
        }

        let mut cursor = macro_idx + "macro_rules!".len();
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        let name_start = cursor;
        while cursor < bytes.len() && is_ident_byte(bytes[cursor]) {
            cursor += 1;
        }
        if name_start == cursor {
            idx = macro_idx + 1;
            continue;
        }
        let name = source[name_start..cursor].to_string();

        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || !matches!(bytes[cursor], b'(' | b'[' | b'{') {
            idx = macro_idx + 1;
            continue;
        }

        let Some(close_idx) = find_matching_delimiter(bytes, cursor) else {
            idx = macro_idx + 1;
            continue;
        };

        let body = &source[cursor + 1..close_idx];
        let Some(kind) = classify_simple_macro_alias(body) else {
            idx = close_idx + 1;
            continue;
        };

        let line_number = 1 + bytes[..macro_idx].iter().filter(|&&b| b == b'\n').count();
        aliases.insert(name, SimpleMacroAlias { line_number, kind });
        idx = close_idx + 1;
    }

    aliases
}

fn classify_simple_macro_alias(body: &str) -> Option<SimpleMacroAliasKind> {
    let normalized = normalize_no_whitespace(body);
    let placeholder = extract_macro_expr_placeholder(&normalized)?;
    let arrow_idx = normalized.find("=>")?;
    let expansion = normalized[arrow_idx + 2..].trim_end_matches(';');
    let expansion = strip_outer_groups(strip_outer_groups(expansion));

    if expansion == placeholder || expansion == format!("({placeholder})") {
        return Some(SimpleMacroAliasKind::IdentityExpr);
    }
    if expansion.contains("black_box") && expansion.contains(&placeholder) {
        return Some(SimpleMacroAliasKind::BlackBoxTouch);
    }
    if expansion.contains("size_of_val") && expansion.contains(&placeholder) {
        return Some(SimpleMacroAliasKind::SizeOfValTouch);
    }

    None
}

fn extract_macro_expr_placeholder(normalized: &str) -> Option<String> {
    let dollar_idx = normalized.find('$')?;
    let after_dollar = &normalized[dollar_idx + 1..];
    let ident_end = after_dollar
        .find(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .unwrap_or(after_dollar.len());
    if ident_end == 0 {
        return None;
    }
    let ident = &after_dollar[..ident_end];
    let after_ident = &after_dollar[ident_end..];
    if !after_ident.starts_with(":expr") {
        return None;
    }
    Some(format!("${ident}"))
}

fn collect_simple_bool_helpers(source: &str) -> HashMap<String, SimpleBoolHelper> {
    let mut helpers = HashMap::new();
    let bytes = source.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        let Some(rel) = source[idx..].find("fn ") else {
            break;
        };
        let fn_idx = idx + rel;
        if fn_idx > 0 && is_ident_byte(bytes[fn_idx - 1]) {
            idx = fn_idx + 1;
            continue;
        }

        let mut cursor = fn_idx + 3;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        let name_start = cursor;
        while cursor < bytes.len() && is_ident_byte(bytes[cursor]) {
            cursor += 1;
        }
        if name_start == cursor {
            idx = fn_idx + 1;
            continue;
        }
        let name = source[name_start..cursor].to_string();

        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] != b'(' {
            idx = fn_idx + 1;
            continue;
        }
        let Some(params_end) = find_matching_delimiter(bytes, cursor) else {
            idx = fn_idx + 1;
            continue;
        };
        let params = &source[cursor + 1..params_end];
        cursor = params_end + 1;

        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let mut signature_tail = String::new();
        while cursor < bytes.len() {
            if bytes[cursor] == b'{' {
                break;
            }
            signature_tail.push(bytes[cursor] as char);
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] != b'{' {
            idx = fn_idx + 1;
            continue;
        }
        if !normalize_no_whitespace(&signature_tail).contains("->bool") {
            idx = cursor + 1;
            continue;
        }

        let body_start = cursor;
        let Some(body_end) = find_matching_delimiter(bytes, body_start) else {
            idx = fn_idx + 1;
            continue;
        };
        let body = &source[body_start + 1..body_end];
        let Some(body_expr) = extract_helper_body_expression(body) else {
            idx = body_end + 1;
            continue;
        };

        helpers.insert(
            name,
            SimpleBoolHelper {
                params: extract_parameter_names(params),
                body_expr,
            },
        );
        idx = body_end + 1;
    }

    helpers
}

fn extract_helper_body_expression(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix("return ") {
        return Some(rest.trim_end_matches(';').trim().to_string());
    }
    if !trimmed.contains(';') {
        return Some(trimmed.to_string());
    }
    None
}

fn extract_parameter_names(params: &str) -> Vec<String> {
    let mut names = Vec::new();
    for component in split_top_level_components(params) {
        let trimmed = component.trim();
        if trimmed.is_empty() || trimmed.starts_with('&') && trimmed.contains("self") {
            continue;
        }
        if trimmed == "self" || trimmed == "&self" || trimmed == "&mut self" {
            continue;
        }
        let Some(colon_idx) = trimmed.find(':') else {
            continue;
        };
        let mut name = trimmed[..colon_idx].trim();
        while let Some(rest) = name.strip_prefix('&') {
            name = rest.trim_start();
        }
        if let Some(rest) = name.strip_prefix("mut ") {
            name = rest.trim_start();
        }
        if let Some(rest) = name.strip_prefix("ref ") {
            name = rest.trim_start();
        }
        if !name.is_empty()
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            names.push(name.to_string());
        }
    }
    names
}

fn normalize_no_whitespace(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn compute_line_brace_depths(source: &str) -> Vec<usize> {
    let sanitized = strip_comments_and_strings_for_tokens(source);
    let mut depths = vec![0usize];
    let mut depth = 0usize;

    for byte in sanitized {
        match byte {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b'\n' => depths.push(depth),
            _ => {}
        }
    }

    depths
}

fn collect_prior_bool_aliases(
    source: &str,
    line_depths: &[usize],
    target_line: usize,
) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    let lines: Vec<&str> = source.lines().collect();
    if target_line == 0 || target_line > lines.len() {
        return aliases;
    }

    let target_depth = *line_depths.get(target_line - 1).unwrap_or(&0);
    let min_line = target_line.saturating_sub(80).max(1);

    for line_number in (min_line..target_line).rev() {
        let depth = *line_depths.get(line_number - 1).unwrap_or(&0);
        if depth > target_depth {
            continue;
        }

        let trimmed = lines[line_number - 1].trim();
        if depth == 0
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("trait ")
                || trimmed.starts_with("mod "))
        {
            break;
        }

        let Some((name, expr)) = parse_simple_let_assignment_line(trimmed) else {
            continue;
        };
        aliases.entry(name).or_insert(expr);
    }

    aliases
}

fn parse_simple_let_assignment_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    let after_let = trimmed.strip_prefix("let ")?.trim_start();
    let after_let = after_let
        .strip_prefix("mut ")
        .unwrap_or(after_let)
        .trim_start();
    let eq_idx = after_let.find('=')?;
    let lhs = after_let[..eq_idx].trim();
    if lhs.contains('(') || lhs.contains('[') || lhs.contains('{') || lhs.contains(',') {
        return None;
    }
    let name = lhs.split(':').next()?.trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return None;
    }
    let expr = after_let[eq_idx + 1..].trim().trim_end_matches(';').trim();
    if expr.is_empty() {
        return None;
    }
    Some((name.to_string(), expr.to_string()))
}

fn expand_assertion_expression(
    expression: &str,
    local_aliases: &HashMap<String, String>,
    helpers: &HashMap<String, SimpleBoolHelper>,
    macros: &HashMap<String, SimpleMacroAlias>,
    depth: usize,
) -> String {
    if depth > 8 {
        return collapse_whitespace(expression);
    }

    let stripped = strip_outer_groups(expression).trim();

    if let Some(alias_expr) = local_aliases.get(stripped) {
        return expand_assertion_expression(alias_expr, local_aliases, helpers, macros, depth + 1);
    }

    if let Some((macro_name, macro_args)) = extract_macro_invocation(stripped) {
        if let Some(alias) = macros.get(&macro_name) {
            let args = split_top_level_components(&macro_args);
            let first_arg = args.first().copied().unwrap_or("").trim();
            let resolved_arg =
                expand_assertion_expression(first_arg, local_aliases, helpers, macros, depth + 1);
            return match alias.kind {
                SimpleMacroAliasKind::IdentityExpr => resolved_arg,
                SimpleMacroAliasKind::BlackBoxTouch => {
                    format!("__macro_black_box_touch__({resolved_arg})")
                }
                SimpleMacroAliasKind::SizeOfValTouch => {
                    format!("__macro_size_of_val_touch__({resolved_arg})")
                }
            };
        }
    }

    if let Some((callee, args_text)) = extract_call_expression(stripped) {
        if let Some(helper) = helpers.get(&callee) {
            let args = split_top_level_components(&args_text);
            if args.len() == helper.params.len() {
                let mut mapping = HashMap::new();
                for (param, arg) in helper.params.iter().zip(args.iter()) {
                    mapping.insert(
                        param.as_str(),
                        expand_assertion_expression(
                            arg.trim(),
                            local_aliases,
                            helpers,
                            macros,
                            depth + 1,
                        ),
                    );
                }
                let substituted = substitute_identifiers(&helper.body_expr, &mapping);
                return expand_assertion_expression(
                    &substituted,
                    local_aliases,
                    helpers,
                    macros,
                    depth + 1,
                );
            }
        }
    }

    let or_parts = split_top_level_boolean(stripped, BooleanJoin::Or);
    if or_parts.len() > 1 {
        return or_parts
            .into_iter()
            .map(|part| {
                format!(
                    "({})",
                    expand_assertion_expression(part, local_aliases, helpers, macros, depth + 1)
                )
            })
            .collect::<Vec<_>>()
            .join(" || ");
    }

    let and_parts = split_top_level_boolean(stripped, BooleanJoin::And);
    if and_parts.len() > 1 {
        return and_parts
            .into_iter()
            .map(|part| {
                format!(
                    "({})",
                    expand_assertion_expression(part, local_aliases, helpers, macros, depth + 1)
                )
            })
            .collect::<Vec<_>>()
            .join(" && ");
    }

    if let Some(rest) = strip_leading_negation(stripped) {
        return format!(
            "!({})",
            expand_assertion_expression(rest, local_aliases, helpers, macros, depth + 1)
        );
    }

    collapse_whitespace(stripped)
}

fn substitute_identifiers(expression: &str, mapping: &HashMap<&str, String>) -> String {
    let mut out = String::with_capacity(expression.len());
    let mut idx = 0usize;
    let bytes = expression.as_bytes();

    while idx < bytes.len() {
        if !is_ident_byte(bytes[idx]) {
            out.push(bytes[idx] as char);
            idx += 1;
            continue;
        }

        let start = idx;
        idx += 1;
        while idx < bytes.len() && is_ident_byte(bytes[idx]) {
            idx += 1;
        }
        let token = &expression[start..idx];
        if let Some(replacement) = mapping.get(token) {
            out.push('(');
            out.push_str(replacement);
            out.push(')');
        } else {
            out.push_str(token);
        }
    }

    out
}

fn detect_resolved_assertion_violation(
    original: &str,
    expanded: &str,
    unsigned_hints: &UnsignedSymbolHints,
) -> Option<String> {
    if expanded.contains("__macro_black_box_touch__") {
        return Some(
            "assertion is routed through a macro wrapper that only performs a black_box touch"
                .to_string(),
        );
    }
    if expanded.contains("__macro_size_of_val_touch__") {
        return Some(
            "assertion is routed through a macro wrapper that only performs a size_of_val touch"
                .to_string(),
        );
    }

    let original_norm = normalize_assertion_expression_for_diff(original);
    let expanded_norm = normalize_assertion_expression_for_diff(expanded);
    let expanded_differs = original_norm != expanded_norm;

    if let Some(reason) = analyze_degenerate_boolean_expression(expanded) {
        if expanded_differs {
            return Some(reason);
        }
    }

    if let Some(reason) = detect_nonnegative_zero_comparison(expanded, unsigned_hints) {
        return Some(reason);
    }

    if let Some(reason) = detect_nonnegative_cmp_matches(expanded, unsigned_hints) {
        return Some(reason);
    }

    None
}

fn normalize_assertion_expression_for_diff(expression: &str) -> String {
    let stripped = strip_outer_groups(expression).trim();

    let or_parts = split_top_level_boolean(stripped, BooleanJoin::Or);
    if or_parts.len() > 1 {
        return or_parts
            .into_iter()
            .map(normalize_assertion_expression_for_diff)
            .collect::<Vec<_>>()
            .join("||");
    }

    let and_parts = split_top_level_boolean(stripped, BooleanJoin::And);
    if and_parts.len() > 1 {
        return and_parts
            .into_iter()
            .map(normalize_assertion_expression_for_diff)
            .collect::<Vec<_>>()
            .join("&&");
    }

    if let Some(rest) = strip_leading_negation(stripped) {
        return format!("!{}", normalize_assertion_expression_for_diff(rest));
    }

    collapse_whitespace(stripped)
}

fn detect_nonnegative_zero_comparison(
    expression: &str,
    unsigned_hints: &UnsignedSymbolHints,
) -> Option<String> {
    let comparison = parse_comparison(expression)?;

    if is_zero_literal(&comparison.right)
        && is_nonnegative_subject(&comparison.left, unsigned_hints)
    {
        return match comparison.op {
            ComparisonOp::Ge => Some(format!(
                "non-negative size/dim/count comparison `{}` is always true",
                format_comparison_expression(&comparison.left, comparison.op, &comparison.right)
            )),
            ComparisonOp::Lt => Some(format!(
                "non-negative size/dim/count comparison `{}` can never be true",
                format_comparison_expression(&comparison.left, comparison.op, &comparison.right)
            )),
            _ => None,
        };
    }

    if is_zero_literal(&comparison.left)
        && is_nonnegative_subject(&comparison.right, unsigned_hints)
    {
        return match comparison.op {
            ComparisonOp::Le => Some(format!(
                "non-negative size/dim/count comparison `{}` is always true",
                format_comparison_expression(&comparison.left, comparison.op, &comparison.right)
            )),
            ComparisonOp::Gt => Some(format!(
                "non-negative size/dim/count comparison `{}` can never be true",
                format_comparison_expression(&comparison.left, comparison.op, &comparison.right)
            )),
            _ => None,
        };
    }

    None
}

fn detect_nonnegative_cmp_matches(
    expression: &str,
    unsigned_hints: &UnsignedSymbolHints,
) -> Option<String> {
    let Some((macro_name, macro_args)) = extract_macro_invocation(expression) else {
        return None;
    };
    if macro_name != "matches" {
        return None;
    }

    let args = split_top_level_components(&macro_args);
    if args.len() < 2 {
        return None;
    }
    let scrutinee = args[0].trim();
    let pattern = args[1].trim();
    let Some(subject) = extract_cmp_zero_subject(scrutinee) else {
        return None;
    };
    if !is_nonnegative_subject(&subject, unsigned_hints) {
        return None;
    }

    let normalized_pattern = normalize_no_whitespace(pattern);
    let has_less =
        normalized_pattern.contains("Ordering::Less") || normalized_pattern.contains("Less");
    let has_equal =
        normalized_pattern.contains("Ordering::Equal") || normalized_pattern.contains("Equal");
    let has_greater =
        normalized_pattern.contains("Ordering::Greater") || normalized_pattern.contains("Greater");

    if has_equal && has_greater && !has_less {
        return Some(
            "matches!(value.cmp(&0), Greater | Equal) on a non-negative size/dim/count is always true"
                .to_string(),
        );
    }
    if has_less && !has_equal && !has_greater {
        return Some(
            "matches!(value.cmp(&0), Less) on a non-negative size/dim/count can never be true"
                .to_string(),
        );
    }

    None
}

fn extract_macro_invocation(expression: &str) -> Option<(String, String)> {
    let stripped = strip_outer_groups(expression).trim();
    let bytes = stripped.as_bytes();
    let bang_idx = stripped.find('!')?;
    let name = stripped[..bang_idx].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == ':')
    {
        return None;
    }

    let mut cursor = bang_idx + 1;
    while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    if cursor >= bytes.len() || !matches!(bytes[cursor], b'(' | b'[' | b'{') {
        return None;
    }

    let close_idx = find_matching_delimiter(bytes, cursor)?;
    if close_idx != bytes.len() - 1 {
        return None;
    }

    Some((
        name.to_string(),
        stripped[cursor + 1..close_idx].trim().to_string(),
    ))
}

fn extract_call_expression(expression: &str) -> Option<(String, String)> {
    let stripped = strip_outer_groups(expression).trim();
    let open_idx = stripped.find('(')?;
    let bytes = stripped.as_bytes();
    let close_idx = find_matching_delimiter(bytes, open_idx)?;
    if close_idx != bytes.len() - 1 {
        return None;
    }
    let callee = stripped[..open_idx].trim();
    if callee.is_empty() {
        return None;
    }
    Some((
        callee.to_string(),
        stripped[open_idx + 1..close_idx].trim().to_string(),
    ))
}

fn extract_cmp_zero_subject(scrutinee: &str) -> Option<String> {
    let (callee, arg) = extract_call_expression(scrutinee)?;
    if !callee.ends_with(".cmp") {
        return None;
    }
    if !is_zero_literal(&arg) {
        return None;
    }
    Some(callee[..callee.len() - ".cmp".len()].trim().to_string())
}

fn is_simple_inert_touch_argument(argument: &str) -> bool {
    let mut candidate = strip_outer_groups(argument).trim();

    loop {
        let stripped = candidate.trim_start();
        if let Some(rest) = stripped.strip_prefix('&') {
            candidate = rest;
            continue;
        }
        if let Some(rest) = stripped.strip_prefix('*') {
            candidate = rest;
            continue;
        }
        if let Some(rest) = stripped.strip_prefix("mut ") {
            candidate = rest;
            continue;
        }
        candidate = stripped;
        break;
    }

    if candidate.is_empty() {
        return false;
    }
    if candidate.contains('(') || candidate.contains('[') || candidate.contains('{') {
        return false;
    }
    if candidate
        .chars()
        .any(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '.')))
    {
        return false;
    }

    candidate.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn collect_unsigned_symbol_hints(source: &str) -> UnsignedSymbolHints {
    let mut hints = UnsignedSymbolHints {
        value_names: HashSet::new(),
        method_names: HashSet::new(),
    };

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(name) = parse_unsigned_method_signature(trimmed) {
            hints.method_names.insert(name);
        }
        if let Some(name) = parse_typed_unsigned_value_name(trimmed) {
            hints.value_names.insert(name);
        }
    }

    for line in source.lines() {
        let trimmed = line.trim();
        let Some((name, expr)) = parse_simple_let_assignment_line(trimmed) else {
            continue;
        };
        if is_obviously_nonnegative_expression(&expr, &hints) {
            hints.value_names.insert(name);
        }
    }

    hints
}

fn parse_unsigned_method_signature(line: &str) -> Option<String> {
    let mut rest = line.trim();
    if let Some(after_vis) = rest.strip_prefix("pub(crate) ") {
        rest = after_vis;
    } else if let Some(after_vis) = rest.strip_prefix("pub ") {
        rest = after_vis;
    }
    if let Some(after_kw) = rest.strip_prefix("unsafe ") {
        rest = after_kw;
    }
    if let Some(after_kw) = rest.strip_prefix("async ") {
        rest = after_kw;
    }
    if let Some(after_kw) = rest.strip_prefix("const ") {
        rest = after_kw;
    }
    let rest = rest.strip_prefix("fn ")?;
    let name_end = rest
        .find(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .unwrap_or(rest.len());
    if name_end == 0 {
        return None;
    }
    let name = &rest[..name_end];
    let return_idx = rest.find("->")?;
    let return_ty = rest[return_idx + 2..].trim();
    if !is_unsigned_type_annotation(return_ty) {
        return None;
    }
    Some(name.to_string())
}

fn parse_typed_unsigned_value_name(line: &str) -> Option<String> {
    let trimmed = line.trim();

    if let Some(after_let) = trimmed.strip_prefix("let ") {
        let after_let = after_let
            .strip_prefix("mut ")
            .unwrap_or(after_let)
            .trim_start();
        let colon_idx = after_let.find(':')?;
        let eq_idx = after_let.find('=')?;
        if colon_idx > eq_idx {
            return None;
        }
        let name = after_let[..colon_idx].trim();
        let ty = after_let[colon_idx + 1..eq_idx].trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            || !is_unsigned_type_annotation(ty)
        {
            return None;
        }
        return Some(name.to_string());
    }

    let after_vis = trimmed
        .strip_prefix("pub(crate) ")
        .or_else(|| trimmed.strip_prefix("pub "))
        .unwrap_or(trimmed);
    let colon_idx = after_vis.find(':')?;
    let name = after_vis[..colon_idx].trim();
    let ty = after_vis[colon_idx + 1..].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        || !is_unsigned_type_annotation(ty)
    {
        return None;
    }
    Some(name.to_string())
}

fn is_unsigned_type_annotation(ty: &str) -> bool {
    let ty = ty
        .trim()
        .trim_end_matches(',')
        .trim_end_matches('{')
        .trim_end_matches(';')
        .trim();
    matches!(ty, "usize" | "u8" | "u16" | "u32" | "u64" | "u128")
}

fn is_obviously_nonnegative_expression(expr: &str, hints: &UnsignedSymbolHints) -> bool {
    let stripped = strip_outer_groups(expr).trim();
    if stripped.ends_with(".len()")
        || stripped.ends_with(".count()")
        || stripped.ends_with(".capacity()")
        || stripped.ends_with(".nrows()")
        || stripped.ends_with(".ncols()")
        || stripped.contains("sum::<usize>()")
    {
        return true;
    }
    is_nonnegative_subject(stripped, hints)
}

fn is_nonnegative_subject(subject: &str, hints: &UnsignedSymbolHints) -> bool {
    let stripped = strip_outer_groups(subject).trim();
    if stripped.ends_with(".len()")
        || stripped.ends_with(".count()")
        || stripped.ends_with(".capacity()")
        || stripped.ends_with(".nrows()")
        || stripped.ends_with(".ncols()")
    {
        return true;
    }

    let Some(name) = extract_terminal_name(stripped) else {
        return false;
    };
    hints.value_names.contains(&name)
        || hints.method_names.contains(&name)
        || looks_size_like_name(&name)
}

fn extract_terminal_name(expression: &str) -> Option<String> {
    let stripped = strip_outer_groups(expression).trim();
    if let Some((callee, _)) = extract_call_expression(stripped) {
        let last = callee
            .rsplit_once("::")
            .map(|(_, rhs)| rhs)
            .unwrap_or(callee.as_str())
            .rsplit_once('.')
            .map(|(_, rhs)| rhs)
            .unwrap_or(callee.as_str());
        if !last.is_empty()
            && last
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return Some(last.to_string());
        }
        return None;
    }

    let candidate = stripped
        .rsplit_once("::")
        .map(|(_, rhs)| rhs)
        .unwrap_or(stripped)
        .rsplit_once('.')
        .map(|(_, rhs)| rhs)
        .unwrap_or(stripped)
        .trim();
    if candidate.is_empty()
        || !candidate
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return None;
    }
    Some(candidate.to_string())
}

fn looks_size_like_name(name: &str) -> bool {
    matches!(name, "len" | "count" | "capacity" | "nrows" | "ncols")
        || name.ends_with("_len")
        || name.ends_with("_count")
        || name.ends_with("_size")
        || name.ends_with("_dim")
}

fn is_zero_literal(expression: &str) -> bool {
    let mut stripped = strip_outer_groups(expression).trim();
    while let Some(rest) = stripped.strip_prefix('&') {
        stripped = rest.trim_start();
    }
    if stripped == "0" {
        return true;
    }
    let Some(rest) = stripped.strip_prefix('0') else {
        return false;
    };
    !rest.is_empty()
        && rest
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn scan_for_fake_usage() -> Vec<String> {
    // Check for "if var.len() ... { return; }" pattern where var is otherwise unused
    let pattern =
        r"if\s+(!?\s*[a-zA-Z_]\w*)\.(?:len|is_empty)\(\)\s*(?:[!=<>]+[^\{]+)?\{\s*return\s*;?\s*\}";
    let matcher = match RegexMatcher::new_line_matcher(pattern) {
        Ok(m) => m,
        Err(e) => return vec![format!("Error creating fake usage regex: {}", e)],
    };

    let mut allviolations = Vec::new();

    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !is_in_ignored_directory(e.path()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Sanitize to avoid matching in comments/strings, and for accurate counting
        let sanitized_bytes = strip_comments_and_strings_for_content(&content);
        let sanitized = String::from_utf8_lossy(&sanitized_bytes);

        let mut collector = FakeUsageCandidateCollector::new();
        if Searcher::new()
            .search_reader(&matcher, Cursor::new(&sanitized_bytes), &mut collector)
            .is_err()
        {
            continue;
        }

        for (var_raw, line_num, line_text) in collector.candidates {
            // Clean up var name (remove !, whitespace)
            let var = var_raw.trim_start_matches('!').trim();

            // Count occurrences in the WHOLE file (sanitized)
            // We use word boundary check
            let count = countvariable_occurrences(&sanitized, var);

            // If count is low (<= 2), it means it's likely only used in signature/let and this check
            // We use a threshold of 2: One for definition/signature, one for the check itself.
            // If it's used anywhere else, count would be >= 3.
            if count <= 2 {
                let filename = path.to_str().unwrap_or("?");
                let mut msg = format!(
                    "\n❌ ERROR: Found fake usage of variable '{}' in {}:{}:\n",
                    var, filename, line_num
                );
                msg.push_str(&format!("   {}\n", line_text));
                msg.push_str("\n⚠️ This looks like a dummy check to mask an unused variable.\n");
                msg.push_str("   \"No-Op Variable Access\": The variable is checked but never referenced again.\n");
                msg.push_str("   Remove the variable (or rename to _) and the check.\n");
                msg.push_str("   These are the only two options: fully wire it into real use, or delete it entirely.\n");
                msg.push_str("   Do not suppress, hide, no-op, stub, or work around this lint in any other way.\n");
                allviolations.push(msg);
            }
        }
    }
    allviolations
}

fn scan_for_degenerate_boolean_expressions() -> Vec<String> {
    let mut allviolations = Vec::new();

    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !is_in_ignored_directory(e.path()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let source = match std::fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(_) => continue,
        };

        let mut collector = DegenerateBooleanCollector::new(path);
        collector
            .violations
            .extend(find_degenerate_boolean_expressions(&source));

        if let Some(error_message) = collector.check_and_get_error_message() {
            allviolations.push(error_message);
        }
    }

    allviolations
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BooleanJoin {
    Or,
    And,
}

impl BooleanJoin {
    fn symbol(self) -> &'static str {
        match self {
            Self::Or => "||",
            Self::And => "&&",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl ComparisonOp {
    fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PredicateKind {
    Some,
    None,
    Ok,
    Err,
}

impl PredicateKind {
    fn is_complement_of(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Some, Self::None)
                | (Self::None, Self::Some)
                | (Self::Ok, Self::Err)
                | (Self::Err, Self::Ok)
        )
    }
}

struct BooleanMacroTarget {
    line_number: usize,
    macro_name: &'static str,
    expression: String,
}

struct ParsedBoolAtom {
    raw: String,
    negated_inner: Option<String>,
    comparison: Option<ParsedComparison>,
    predicate: Option<ParsedPredicate>,
}

struct ParsedComparison {
    left: String,
    op: ComparisonOp,
    right: String,
}

struct ParsedPredicate {
    subject: String,
    kind: PredicateKind,
}

fn find_degenerate_boolean_expressions(source: &str) -> Vec<String> {
    let sanitized_bytes = strip_comments_and_strings_for_content(source);
    let sanitized = String::from_utf8_lossy(&sanitized_bytes);
    let mut violations = Vec::new();

    for target in collect_boolean_macro_targets(&sanitized) {
        let Some(reason) = analyze_degenerate_boolean_expression(&target.expression) else {
            continue;
        };

        let snippet = truncate_for_display(&collapse_whitespace(&target.expression), 160);
        violations.push(format!(
            "{}:{}({}) -> {}",
            target.line_number, target.macro_name, snippet, reason
        ));
    }

    violations
}

fn collect_boolean_macro_targets(source: &str) -> Vec<BooleanMacroTarget> {
    let mut targets = Vec::new();
    collect_macro_first_argument_targets(source, "assert!", &mut targets);
    collect_macro_first_argument_targets(source, "debug_assert!", &mut targets);
    targets
}

fn collect_macro_first_argument_targets(
    source: &str,
    macro_name: &'static str,
    targets: &mut Vec<BooleanMacroTarget>,
) {
    let bytes = source.as_bytes();
    let pattern = macro_name.as_bytes();
    let mut idx = 0usize;

    while idx + pattern.len() <= bytes.len() {
        if &bytes[idx..idx + pattern.len()] != pattern {
            idx += 1;
            continue;
        }

        if idx > 0 && is_ident_byte(bytes[idx - 1]) {
            idx += 1;
            continue;
        }

        let mut cursor = idx + pattern.len();
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        if cursor >= bytes.len() || !matches!(bytes[cursor], b'(' | b'[' | b'{') {
            idx += 1;
            continue;
        }

        let Some(close_idx) = find_matching_delimiter(bytes, cursor) else {
            idx += 1;
            continue;
        };

        let arg_start = cursor + 1;
        let arg_end = find_first_top_level_comma(bytes, arg_start, close_idx).unwrap_or(close_idx);
        let expression = source[arg_start..arg_end].trim();
        if !expression.is_empty() {
            let line_number = 1 + bytes[..idx].iter().filter(|&&b| b == b'\n').count();
            targets.push(BooleanMacroTarget {
                line_number,
                macro_name,
                expression: expression.to_string(),
            });
        }

        idx = close_idx + 1;
    }
}

fn analyze_degenerate_boolean_expression(expression: &str) -> Option<String> {
    analyze_degenerate_boolean_expression_inner(expression, 0)
}

fn analyze_degenerate_boolean_expression_inner(expression: &str, depth: usize) -> Option<String> {
    if depth > 8 {
        return None;
    }

    let stripped = strip_outer_groups(expression);

    for join in [BooleanJoin::Or, BooleanJoin::And] {
        let clauses = split_top_level_boolean(stripped, join);
        if clauses.len() <= 1 {
            continue;
        }

        let atoms: Vec<ParsedBoolAtom> = clauses
            .iter()
            .map(|clause| parse_bool_atom(clause))
            .collect();
        for left_idx in 0..atoms.len() {
            for right_idx in left_idx + 1..atoms.len() {
                if let Some(reason) =
                    diagnose_boolean_pair(join, &atoms[left_idx], &atoms[right_idx])
                {
                    return Some(reason);
                }
            }
        }

        for clause in clauses {
            if let Some(reason) = analyze_degenerate_boolean_expression_inner(clause, depth + 1) {
                return Some(reason);
            }
        }

        return None;
    }

    let without_negation = strip_leading_negation(stripped)?;
    analyze_degenerate_boolean_expression_inner(without_negation, depth + 1)
}

/// Returns true if the normalized expression contains a masked string literal
/// (a run of 'x' characters that replaced the original string content during
/// `strip_comments_and_strings_for_content`).  Two atoms that differ only in
/// their string-literal contents will look identical after masking, so we must
/// skip the raw-equality duplicate check when masked strings are present.
fn contains_masked_string_literal(normalized: &str) -> bool {
    // After normalization (whitespace stripped), a masked string literal shows
    // up as a run of 3+ consecutive 'x' characters (quote + at least one
    // content char + quote, all mapped to 'x').  We use 3 as the threshold to
    // avoid matching short identifiers like "xx".
    let mut consecutive = 0u32;
    for ch in normalized.chars() {
        if ch == 'x' {
            consecutive += 1;
            if consecutive >= 3 {
                return true;
            }
        } else {
            consecutive = 0;
        }
    }
    false
}

fn diagnose_boolean_pair(
    join: BooleanJoin,
    left: &ParsedBoolAtom,
    right: &ParsedBoolAtom,
) -> Option<String> {
    if left.raw == right.raw && !contains_masked_string_literal(&left.raw) {
        return Some(format!(
            "duplicate clause repeated on both sides of `{}`",
            join.symbol()
        ));
    }

    if left.negated_inner.as_deref() == Some(right.raw.as_str())
        || right.negated_inner.as_deref() == Some(left.raw.as_str())
    {
        return Some(match join {
            BooleanJoin::Or => "boolean complement pair is always true".to_string(),
            BooleanJoin::And => "boolean complement pair can never be true".to_string(),
        });
    }

    if let (Some(left_predicate), Some(right_predicate)) = (&left.predicate, &right.predicate) {
        if left_predicate.subject == right_predicate.subject
            && left_predicate.kind.is_complement_of(right_predicate.kind)
        {
            return Some(match join {
                BooleanJoin::Or => "complementary predicate pair is always true".to_string(),
                BooleanJoin::And => "complementary predicate pair can never be true".to_string(),
            });
        }
    }

    let (Some(left_comparison), Some(right_comparison)) = (&left.comparison, &right.comparison)
    else {
        return None;
    };

    let Some((left_operand, right_operand)) =
        shared_comparison_operands(left_comparison, right_comparison)
    else {
        return None;
    };

    let result_mask = match join {
        BooleanJoin::Or => {
            comparison_mask(left_comparison.op) | comparison_mask(right_comparison.op)
        }
        BooleanJoin::And => {
            comparison_mask(left_comparison.op) & comparison_mask(right_comparison.op)
        }
    };

    if result_mask == 0 {
        return Some("comparison pair can never be true".to_string());
    }
    if result_mask == 0b111 {
        return Some("comparison pair is always true".to_string());
    }

    let equivalent_op = comparison_from_mask(result_mask)?;
    let equivalent = format_comparison_expression(&left_operand, equivalent_op, &right_operand);
    Some(format!("comparison pair collapses to `{equivalent}`"))
}

fn parse_bool_atom(expression: &str) -> ParsedBoolAtom {
    let stripped = strip_outer_groups(expression);
    let raw = normalize_boolean_text(stripped);
    let negated_inner = strip_leading_negation(stripped).map(normalize_boolean_text);
    let comparison = parse_comparison(stripped);
    let predicate = parse_complementary_predicate(&raw);

    ParsedBoolAtom {
        raw,
        negated_inner,
        comparison,
        predicate,
    }
}

fn parse_comparison(expression: &str) -> Option<ParsedComparison> {
    let stripped = strip_outer_groups(expression);
    let bytes = stripped.as_bytes();
    let mut stack = Vec::new();
    let mut idx = 0usize;
    let mut found: Option<(usize, ComparisonOp, usize)> = None;

    while idx < bytes.len() {
        match bytes[idx] {
            b'(' | b'[' | b'{' => {
                stack.push(bytes[idx]);
                idx += 1;
                continue;
            }
            b')' | b']' | b'}' => {
                if stack.pop().is_none() {
                    return None;
                }
                idx += 1;
                continue;
            }
            _ => {}
        }

        if !stack.is_empty() {
            idx += 1;
            continue;
        }

        let candidate = if bytes[idx..].starts_with(b"==") {
            Some((ComparisonOp::Eq, 2))
        } else if bytes[idx..].starts_with(b"!=") {
            Some((ComparisonOp::Ne, 2))
        } else if bytes[idx..].starts_with(b"<=") {
            Some((ComparisonOp::Le, 2))
        } else if bytes[idx..].starts_with(b">=") {
            Some((ComparisonOp::Ge, 2))
        } else if bytes[idx] == b'<' {
            Some((ComparisonOp::Lt, 1))
        } else if bytes[idx] == b'>' {
            Some((ComparisonOp::Gt, 1))
        } else {
            None
        };

        let Some((op, width)) = candidate else {
            idx += 1;
            continue;
        };

        if found.is_some() {
            return None;
        }

        found = Some((idx, op, width));
        idx += width;
    }

    let (operator_idx, operator, operator_width) = found?;
    let left = normalize_boolean_text(&stripped[..operator_idx]);
    let right = normalize_boolean_text(&stripped[operator_idx + operator_width..]);
    if left.is_empty() || right.is_empty() {
        return None;
    }

    Some(ParsedComparison {
        left,
        op: operator,
        right,
    })
}

fn parse_complementary_predicate(normalized_expression: &str) -> Option<ParsedPredicate> {
    const PREDICATE_SUFFIXES: [(&str, PredicateKind); 4] = [
        (".is_some()", PredicateKind::Some),
        (".is_none()", PredicateKind::None),
        (".is_ok()", PredicateKind::Ok),
        (".is_err()", PredicateKind::Err),
    ];

    for (suffix, kind) in PREDICATE_SUFFIXES {
        let Some(subject) = normalized_expression.strip_suffix(suffix) else {
            continue;
        };
        if subject.is_empty() {
            continue;
        }
        return Some(ParsedPredicate {
            subject: subject.to_string(),
            kind,
        });
    }

    None
}

fn shared_comparison_operands(
    left: &ParsedComparison,
    right: &ParsedComparison,
) -> Option<(String, String)> {
    if left.left == right.left && left.right == right.right {
        return Some((left.left.clone(), left.right.clone()));
    }

    if matches!(left.op, ComparisonOp::Eq | ComparisonOp::Ne)
        && matches!(right.op, ComparisonOp::Eq | ComparisonOp::Ne)
        && left.left == right.right
        && left.right == right.left
    {
        return if left.left <= left.right {
            Some((left.left.clone(), left.right.clone()))
        } else {
            Some((left.right.clone(), left.left.clone()))
        };
    }

    None
}

fn comparison_mask(operator: ComparisonOp) -> u8 {
    match operator {
        ComparisonOp::Eq => 0b010,
        ComparisonOp::Ne => 0b101,
        ComparisonOp::Lt => 0b001,
        ComparisonOp::Le => 0b011,
        ComparisonOp::Gt => 0b100,
        ComparisonOp::Ge => 0b110,
    }
}

fn comparison_from_mask(mask: u8) -> Option<ComparisonOp> {
    match mask {
        0b001 => Some(ComparisonOp::Lt),
        0b010 => Some(ComparisonOp::Eq),
        0b011 => Some(ComparisonOp::Le),
        0b100 => Some(ComparisonOp::Gt),
        0b101 => Some(ComparisonOp::Ne),
        0b110 => Some(ComparisonOp::Ge),
        _ => None,
    }
}

fn format_comparison_expression(left: &str, operator: ComparisonOp, right: &str) -> String {
    format!("{left} {} {right}", operator.symbol())
}

fn split_top_level_boolean(expression: &str, join: BooleanJoin) -> Vec<&str> {
    let bytes = expression.as_bytes();
    let mut parts = Vec::new();
    let mut stack = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    let symbol = join.symbol().as_bytes();

    while idx < bytes.len() {
        match bytes[idx] {
            b'(' | b'[' | b'{' => stack.push(bytes[idx]),
            b')' | b']' | b'}' => {
                stack.pop();
            }
            _ => {}
        }

        if stack.is_empty()
            && idx + symbol.len() <= bytes.len()
            && &bytes[idx..idx + symbol.len()] == symbol
        {
            parts.push(expression[start..idx].trim());
            start = idx + symbol.len();
            idx += symbol.len();
            continue;
        }

        idx += 1;
    }

    if start == 0 {
        return vec![expression.trim()];
    }

    parts.push(expression[start..].trim());
    parts.retain(|part| !part.is_empty());
    parts
}

fn strip_outer_groups(mut expression: &str) -> &str {
    loop {
        let trimmed = expression.trim();
        let bytes = trimmed.as_bytes();
        if bytes.len() < 2 || !matches!(bytes[0], b'(' | b'[' | b'{') {
            return trimmed;
        }

        let Some(close_idx) = find_matching_delimiter(bytes, 0) else {
            return trimmed;
        };
        if close_idx != bytes.len() - 1 {
            return trimmed;
        }

        expression = &trimmed[1..trimmed.len() - 1];
    }
}

fn strip_leading_negation(expression: &str) -> Option<&str> {
    let trimmed = expression.trim_start();
    let remainder = trimmed.strip_prefix('!')?;
    Some(strip_outer_groups(remainder))
}

fn normalize_boolean_text(expression: &str) -> String {
    strip_outer_groups(expression)
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect()
}

fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::new();
    let mut saw_whitespace = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            saw_whitespace = true;
            continue;
        }

        if saw_whitespace && !collapsed.is_empty() {
            collapsed.push(' ');
        }
        saw_whitespace = false;
        collapsed.push(ch);
    }

    collapsed
}

fn truncate_for_display(text: &str, max_len: usize) -> String {
    let mut truncated = String::new();
    for ch in text.chars() {
        if truncated.len() + ch.len_utf8() > max_len {
            truncated.push_str("...");
            return truncated;
        }
        truncated.push(ch);
    }
    truncated
}

fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn find_matching_delimiter(bytes: &[u8], open_idx: usize) -> Option<usize> {
    let mut stack = Vec::new();

    for (offset, byte) in bytes.iter().enumerate().skip(open_idx) {
        match *byte {
            b'(' | b'[' | b'{' => stack.push(*byte),
            b')' | b']' | b'}' => {
                let open = stack.pop()?;
                if !delimiters_match(open, *byte) {
                    return None;
                }
                if stack.is_empty() {
                    return Some(offset);
                }
            }
            _ => {}
        }
    }

    None
}

fn delimiters_match(open: u8, close: u8) -> bool {
    matches!((open, close), (b'(', b')') | (b'[', b']') | (b'{', b'}'))
}

fn find_first_top_level_comma(bytes: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut stack = Vec::new();
    let mut idx = start;

    while idx < end {
        match bytes[idx] {
            b'(' | b'[' | b'{' => stack.push(bytes[idx]),
            b')' | b']' | b'}' => {
                if stack.pop().is_none() {
                    return None;
                }
            }
            b',' if stack.is_empty() => return Some(idx),
            _ => {}
        }
        idx += 1;
    }

    None
}

struct FakeUsageCandidateCollector {
    candidates: Vec<(String, u64, String)>, // var_raw, line_num, line_text
}

impl FakeUsageCandidateCollector {
    fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }
}

impl Sink for FakeUsageCandidateCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim();

        // Extract variable name manually since we can't get capture groups easily
        // Pattern: if (!? var) . (len|is_empty)
        if let Some(if_idx) = line_text.find("if") {
            let after_if = &line_text[if_idx + 2..];
            if let Some(dot_idx) = after_if.find('.') {
                let var_part = &after_if[..dot_idx];
                self.candidates.push((
                    var_part.trim().to_string(),
                    line_number,
                    line_text.to_string(),
                ));
            }
        }

        Ok(true)
    }
}

fn countvariable_occurrences(text: &str, var: &str) -> usize {
    let mut count = 0;
    for (idx, _) in text.match_indices(var) {
        if isword_boundary(text, idx, var.len()) {
            count += 1;
        }
    }
    count
}

fn isword_boundary(text: &str, idx: usize, len: usize) -> bool {
    let before = if idx == 0 {
        None
    } else {
        text[..idx].chars().last()
    };
    let after = text[idx + len..].chars().next();

    let isword_char = |c: char| c.is_alphanumeric() || c == '_';

    if before.is_some_and(isword_char) {
        return false;
    }
    if after.is_some_and(isword_char) {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Dead public item detection
//
// Scans for `pub` and `pub(crate)` items that have no real production
// references.  Designed to resist evasion by automated agents:
//
//   1. `pub` -> `pub(crate)` downgrade: both visibilities are scanned.
//   2. Fake callers (`let _ = foo;`, `if false { foo(); }`): filtered out.
//   3. Re-export padding (`pub use`): filtered out.
//   4. Noop touch (`name;`): filtered out.
//   5. Test-only usage: references in tests/ and #[cfg(test)] blocks are
//      counted separately; items alive only in tests get a distinct error
//      telling the author to move the item to #[cfg(test)] or wire it
//      into production code.
// ---------------------------------------------------------------------------

/// Visibility level extracted from a definition line.
#[derive(Clone, Copy, PartialEq)]
enum PubVisibility {
    Pub,
    PubCrate,
}

/// Stripped file contents for dead-code analysis.
struct StrippedFile {
    path: PathBuf,
    /// Full comment/string-stripped source.
    stripped: String,
    /// `true` when the file lives under `tests/` or `benches/`.
    is_test_file: bool,
}

/// Walk all `.rs` files (excluding build.rs and ignored directories), read each,
/// strip comments and string literals, and return.
fn build_stripped_file_cache() -> Vec<StrippedFile> {
    let mut files = Vec::new();
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !is_in_ignored_directory(e.path()))
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path().to_path_buf();
        if path.file_name().is_some_and(|f| f == "build.rs") {
            continue;
        }
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let stripped_bytes = strip_comments_and_strings_for_content(&content);
        let stripped = String::from_utf8_lossy(&stripped_bytes).to_string();

        let is_test_file = is_test_or_bench_path(&path);

        files.push(StrippedFile {
            path,
            stripped,
            is_test_file,
        });
    }
    files
}

/// Returns `true` for files under `tests/`, `benches/`, or test helper crates.
fn is_test_or_bench_path(path: &Path) -> bool {
    let s = path.to_string_lossy();
    s.contains("/tests/")
        || s.contains("/benches/")
        || s.starts_with("./tests/")
        || s.starts_with("./benches/")
}

/// Try to extract a public item name from a single (stripped) source line.
///
/// Matches both `pub` and `pub(crate)` items.  Returns the name and visibility.
/// Skips `pub(super)` and `pub(in ...)` — those are too narrow to be evasion.
fn extract_pub_item_name(line: &str) -> Option<(&str, PubVisibility)> {
    let trimmed = line.trim();

    // Determine visibility.
    let (after_vis, visibility) = if trimmed.starts_with("pub(crate)") {
        let rest = trimmed["pub(crate)".len()..].trim_start();
        (rest, PubVisibility::PubCrate)
    } else if trimmed.starts_with("pub(") {
        // pub(super), pub(in ...) — skip entirely.
        return None;
    } else if let Some(rest) = trimmed.strip_prefix("pub ") {
        (rest, PubVisibility::Pub)
    } else {
        return None;
    };

    // Strip optional qualifiers between visibility and keyword.
    let rest = after_vis
        .strip_prefix("unsafe ")
        .or_else(|| after_vis.strip_prefix("async "))
        .or_else(|| after_vis.strip_prefix("const "))
        .unwrap_or(after_vis);

    // Must start with a defining keyword.
    let after_kw = rest
        .strip_prefix("fn ")
        .or_else(|| rest.strip_prefix("struct "))
        .or_else(|| rest.strip_prefix("enum "))
        .or_else(|| rest.strip_prefix("type "))?;

    // Extract the identifier.
    let end = after_kw
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(after_kw.len());
    let name = &after_kw[..end];
    if name.is_empty() {
        return None;
    }
    Some((name, visibility))
}

/// Count references to `name` in `text`, excluding fake / evasive patterns.
///
/// Filtered patterns (per line):
///   - `let _ = name` / `let mut _ = name` (discard binding)
///   - `_ = name` (bare discard assignment)
///   - `if false { ... name ... }` (dead branch)
///   - `pub use ... name` / `pub(crate) use ...` (re-export padding)
///   - bare `name;` on its own line (noop touch)
fn count_real_references(text: &str, name: &str) -> usize {
    let mut real_count: usize = 0;

    for line in text.lines() {
        let hits = countvariable_occurrences(line, name);
        if hits == 0 {
            continue;
        }
        let trimmed = line.trim();

        // Filter: discard binding — `let _ = name` or `let mut _ = name`
        if is_discard_binding(trimmed, name) {
            continue;
        }

        // Filter: bare discard — `_ = name(...)`
        if trimmed.starts_with("_ =") {
            let after_eq = trimmed[2..].trim_start_matches('=').trim_start();
            if after_eq.starts_with(name) {
                continue;
            }
        }

        // Filter: dead branch — `if false {`
        if trimmed.starts_with("if false") {
            continue;
        }

        // Filter: re-export padding — `pub use`, `pub(crate) use`
        if is_reexport_line(trimmed) {
            continue;
        }

        // Filter: noop touch — bare `name;` or just `name`
        if trimmed == name || trimmed.strip_suffix(';').is_some_and(|s| s.trim() == name) {
            continue;
        }

        real_count += hits;
    }

    real_count
}

/// Check if this line is a discard binding for the target name.
/// Matches: `let _ = name`, `let _ = name(`, `let _ = name;`
fn is_discard_binding(trimmed: &str, name: &str) -> bool {
    let after_let = match trimmed.strip_prefix("let ") {
        Some(s) => s.trim_start(),
        None => return false,
    };
    // Must start with `_` and then `=` (possibly with `mut` in between).
    if !after_let.starts_with('_') {
        return false;
    }
    let after_underscore = if let Some(rest) = after_let.strip_prefix("mut _") {
        rest
    } else {
        &after_let[1..]
    };
    // Next non-whitespace must be `=` (not `==`).
    let after_ws = after_underscore.trim_start();
    if !after_ws.starts_with('=') || after_ws.starts_with("==") {
        return false;
    }
    let rhs = after_ws[1..].trim_start();
    // RHS must reference our target name.
    if !rhs.starts_with(name) {
        return false;
    }
    let after_name = &rhs[name.len()..];
    after_name.is_empty()
        || after_name.starts_with(';')
        || after_name.starts_with('(')
        || after_name.starts_with('.')
        || after_name.starts_with(' ')
        || after_name.starts_with(':')
}

/// Check if the line is a `pub use` or `pub(crate) use` re-export.
fn is_reexport_line(trimmed: &str) -> bool {
    trimmed.starts_with("pub use ")
        || trimmed.starts_with("pub(crate) use ")
        || trimmed.starts_with("pub(super) use ")
}

/// Core dead-public-item scanner.
///
/// For each `pub` / `pub(crate)` item defined in production code (src/, outside
/// `#[cfg(test)]`), counts real (non-fake) references in production zones
/// vs test zones separately, then categorizes:
///
///   - **Dead everywhere**: zero real references in both prod and test.
///   - **Dead in production**: zero prod refs, but test refs exist —
///     tests are testing dead code; item should be `#[cfg(test)]` or wired
///     into prod.
///   - **pub(crate) downgrade**: item was made `pub(crate)` but is still dead.
fn scan_for_dead_public_items(cache: &[StrippedFile]) -> Vec<String> {
    use std::collections::HashSet;

    let mut allviolations = Vec::new();

    // ---- Git-tiered enforcement ----
    // Uncommitted files → silent (skip entirely, WIP grace period).
    // Committed but not pushed → warn only (nudge, don't block).
    // Pushed → error (fail the build).
    fn git_changed_files(args: &[&str]) -> HashSet<String> {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(|l| format!("./{l}"))
                    .collect()
            })
            .unwrap_or_default()
    }
    let mut uncommitted_files = git_changed_files(&["diff", "--name-only", "HEAD"]);
    uncommitted_files.extend(git_changed_files(&[
        "ls-files",
        "--others",
        "--exclude-standard",
    ]));
    let unpushed_files = git_changed_files(&["diff", "--name-only", "@{upstream}..HEAD"]);

    // Names to skip: common trait methods and very short / very common identifiers
    // that would collide with unrelated identifiers producing false positives.
    let skip_names: HashSet<&str> = [
        // Trait methods
        "new",
        "default",
        "from",
        "into",
        "fmt",
        "clone",
        "drop",
        "eq",
        "ne",
        "cmp",
        "partial_cmp",
        "hash",
        "deref",
        "deref_mut",
        "index",
        "index_mut",
        "next",
        "size_hint",
        "try_from",
        "try_into",
        "serialize",
        "deserialize",
        "matched",
        "main",
        // Too short / too common — high collision risk
        "run",
        "get",
        "set",
        "add",
        "len",
    ]
    .iter()
    .copied()
    .collect();

    struct PubItem {
        name: String,
        file: String,
        line: usize,
        visibility: PubVisibility,
    }
    let mut pub_items: Vec<PubItem> = Vec::new();

    // ---- Phase 1: extract definitions from production zones of src/ files ----
    for sf in cache {
        if sf.is_test_file {
            continue; // Don't scan test files for dead definitions.
        }
        let file_str = sf.path.to_string_lossy().to_string();
        if uncommitted_files.contains(&file_str) {
            continue; // WIP file — silent grace period.
        }
        // Extract definitions from stripped source lines (1-based line numbers).
        for (line_idx, line) in sf.stripped.lines().enumerate() {
            if let Some((name, vis)) = extract_pub_item_name(line) {
                if name.len() >= 4 && !skip_names.contains(name) {
                    pub_items.push(PubItem {
                        name: name.to_string(),
                        file: sf.path.to_string_lossy().to_string(),
                        line: line_idx + 1,
                        visibility: vis,
                    });
                }
            }
        }
    }

    // ---- Phase 2: count prod refs vs test refs with fake-reference filtering ----
    for item in &pub_items {
        let mut prod_refs: usize = 0;
        let mut test_refs: usize = 0;

        // Count references using full stripped text per file.
        // File-level counting avoids fragile brace-depth zone splitting.
        for sf in cache {
            let refs = count_real_references(&sf.stripped, &item.name);
            if sf.is_test_file {
                test_refs += refs;
            } else {
                prod_refs += refs;
            }
        }

        // prod_refs includes the definition itself (count 1).
        // If prod_refs <= 1, no production code calls this item.
        if prod_refs > 1 {
            continue; // Alive in production — not dead.
        }

        // `pub` items used from integration tests (tests/*.rs) are legitimate —
        // that's how Rust's test architecture works.  Only flag `pub` items that
        // have ZERO references anywhere, or `pub(crate)` items (which can't be
        // reached from tests/ anyway, so test refs are in-file #[cfg(test)]).
        if item.visibility == PubVisibility::Pub && test_refs > 0 {
            continue; // Exported for integration tests — not dead.
        }

        // ---- Phase 3: categorize the violation ----
        let vis_label = match item.visibility {
            PubVisibility::Pub => "pub",
            PubVisibility::PubCrate => "pub(crate)",
        };

        let (category, guidance) = if item.visibility == PubVisibility::PubCrate {
            (
                "VISIBILITY DOWNGRADE EVASION",
                "This item was changed to pub(crate) but is still dead.\n   \
                 Reducing visibility does NOT constitute use.\n   \
                 Wire this into production code, or delete it.\n   \
                 These are the only two options: fully wire it into real use, or delete it entirely.\n   \
                 Do not suppress, hide, no-op, stub, or work around this lint in any other way.",
            )
        } else if test_refs > 0 {
            (
                "DEAD IN PRODUCTION (test-only usage)",
                "This item has no production callers \u{2014} it is only referenced from tests.\n   \
                 Tests that test dead code are worse than no tests (they create false confidence).\n   \
                 Either:\n   \
                   \u{2022} Wire the item into production code (preferred), OR\n   \
                   \u{2022} Move it to #[cfg(test)] if it is genuinely a test helper, OR\n   \
                   \u{2022} Delete both the item and its tests.\n   \
                 These are the only options. Fully wire it into real use, or delete it entirely.\n   \
                 Do not suppress, hide, no-op, stub, or work around this lint in any other way.",
            )
        } else {
            (
                "DEAD EVERYWHERE",
                "This item has no references anywhere in the codebase.\n   \
                 Wire it into production code, or delete it.\n   \
                 These are the only two options: fully wire it into real use, or delete it entirely.\n   \
                 Do not suppress, hide, no-op, stub, or work around this lint in any other way.",
            )
        };

        // Tiered enforcement: committed-not-pushed → warn, pushed → error.
        let is_unpushed = unpushed_files.contains(&item.file);
        let (icon, severity) = if is_unpushed {
            ("\u{26a0}\u{fe0f}", "WARNING")
        } else {
            ("\u{274c}", "ERROR")
        };

        let mut msg = format!(
            "\n{icon} {severity}: {category} \u{2014} `{vis}` item '{name}' at {file}:{line}",
            vis = vis_label,
            name = item.name,
            file = item.file,
            line = item.line,
        );
        msg.push_str(&format!("\n   {guidance}"));
        if test_refs > 0 && item.visibility == PubVisibility::Pub {
            msg.push_str(&format!(
                "\n   (Found {} test-only reference(s) that do not count as production use.)",
                test_refs
            ));
        }
        if is_unpushed {
            // Warnings don't block the build — just print them.
            eprintln!("{msg}");
        } else {
            allviolations.push(msg);
        }
    }

    allviolations
}

/// Enforce that `use opt::` imports only appear in strategy.rs.
/// All optimizer access must go through `run_outer` in strategy.rs.
fn scan_for_direct_opt_imports() -> Vec<String> {
    let allowed_files: &[&str] = &["src/solver/strategy.rs"];
    let mut allviolations = Vec::new();

    for entry in WalkDir::new("src")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let path_str = path.to_string_lossy();

        if allowed_files
            .iter()
            .any(|allowed| path_str.ends_with(allowed))
        {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut violation_lines = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            // Skip comments
            if trimmed.starts_with("//") || trimmed.starts_with('*') || trimmed.starts_with("/*") {
                continue;
            }
            if trimmed.contains("use opt::") {
                violation_lines.push(format!("  line {}: {}", line_num + 1, trimmed));
            }
        }

        if !violation_lines.is_empty() {
            allviolations.push(format!(
                "\n\u{274c} ERROR: Direct `use opt::` import in {}.\n\
                 All optimizer access must go through `run_outer` in strategy.rs.\n\
                 {}\n",
                path.display(),
                violation_lines.join("\n"),
            ));
        }
    }

    allviolations
}

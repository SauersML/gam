use grep::regex::RegexMatcher;
use grep::searcher::{Searcher, Sink, SinkMatch};
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

// A custom collector for checking if comments are primarily composed of dashes
struct DashHeavyCommentCollector {
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

// A custom collector for forbidden TODO/FIXME/"for now" comments
struct TodoCommentCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for meaningless conditionals where both branches are identical
struct MeaninglessConditionalCollector {
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
        let _ = io::stdout().flush();
    }
}

fn emit_stage_detail(detail: &str) {
    if warnings_enabled() {
        println!("cargo:warning=project build detail: {detail}");
        let _ = io::stdout().flush();
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

impl DashHeavyCommentCollector {
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
            "\n❌ ERROR: Found {} comments composed primarily of dashes in {}:\n",
            self.violations.len(),
            filename
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Comments where over 80% of non-whitespace characters are dashes are STRICTLY FORBIDDEN in this project.\n",
        );
        error_msg.push_str("   Remove decorative dash-only comments completely.\n");

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
            "\n❌ ERROR: Found {} forbidden 'allow(unused...)' attributes in {}:\n",
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
        error_msg.push_str("   - #[allow(unusedvariables)]\n");
        error_msg.push_str(
            "\n   Do not suppress these warnings. Delete or use the unused code/imports instead.\n",
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

// Implement the `Sink` trait for our collector.
// The `matched` method is called by the searcher for every line that matches the regex.
impl Sink for ViolationCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        // Get the line number and the content of the matched line.
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Skip matches in comments and string literals to avoid false positives
        // But make sure we don't miss underscore variables in code

        // Check if this line is purely a comment
        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        // Check if the match is in a string literal and not part of code
        let mut is_in_string = false;
        if line_text.contains("\"") {
            // More careful string detection logic
            let parts: Vec<&str> = line_text.split('\"').collect();
            // If the underscore variable is between quotes, it's in a string
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("_") {
                    // Inside quotes
                    is_in_string = true;
                    break;
                }
            }
        }

        if is_pure_comment || is_in_string {
            return Ok(true); // Skip this match and continue searching
        }

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

        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        let mut is_in_string = false;
        if line_text.contains("\"") {
            let parts: Vec<&str> = line_text.split('\"').collect();
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("_") {
                    is_in_string = true;
                    break;
                }
            }
        }

        if is_pure_comment || is_in_string {
            return Ok(true);
        }

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
        if trimmed.starts_with("return ") || trimmed.starts_with("return(") {
            return Ok(true);
        }
        if trimmed
            .chars()
            .find(|c| !c.is_whitespace())
            .is_some_and(|c| !c.is_ascii_alphabetic() && c != '_')
        {
            return Ok(true);
        }

        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

impl Sink for TupleWildcardCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        let mut is_in_string = false;
        if line_text.contains("\"") {
            let parts: Vec<&str> = line_text.split('\"').collect();
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("_") {
                    is_in_string = true;
                    break;
                }
            }
        }

        if is_pure_comment || is_in_string {
            return Ok(true);
        }

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

        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        let mut is_in_string = false;
        if line_text.contains("\"") {
            let parts: Vec<&str> = line_text.split('\"').collect();
            let part_count = parts.len();
            for i in 0..part_count {
                if i % 2 == 1 {
                    is_in_string = true;
                    break;
                }
            }
        }

        if is_pure_comment || is_in_string {
            return Ok(true);
        }

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
            ')' | ']' | '}' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
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

impl Sink for DashHeavyCommentCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        if !line_text.trim_start().starts_with("//")
            && !line_text.contains("/*")
            && !line_text.starts_with("///")
        {
            return Ok(true);
        }

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
            return Ok(true);
        };

        let nonwhitespace_chars: Vec<char> = comment_text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();

        if !nonwhitespace_chars.is_empty() {
            let dash_count = nonwhitespace_chars.iter().filter(|c| **c == '-').count();
            let dash_ratio = dash_count as f64 / nonwhitespace_chars.len() as f64;

            if dash_ratio > 0.8 {
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

        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        let mut is_in_string = false;
        if line_text.contains("\"") {
            let parts: Vec<&str> = line_text.split('\"').collect();
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("drop(") {
                    is_in_string = true;
                    break;
                }
            }
        }

        let is_drop_definition = line_text.contains("fn drop")
            || line_text.contains("impl Drop")
            || line_text.contains("trait Drop");

        if is_pure_comment || is_in_string || is_drop_definition {
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

        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        let mut is_in_string = false;
        if line_text.contains("\"") {
            let parts: Vec<&str> = line_text.split('\"').collect();
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("debug_assert!") {
                    is_in_string = true;
                    break;
                }
            }
        }

        if is_pure_comment || is_in_string {
            return Ok(true);
        }

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
        return;
    }

    // Manually check for unused variables in the build script
    update_stage("manual lint self-check");
    manually_check_for_unusedvariables();

    // Collect all violations from all checks
    let mut allviolations = Vec::new();

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

    // Scan for TODO/FIXME/"for now" comments
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

    // Scan for fake usage patterns (dummy checks masking unused variables)
    update_stage("scan fake usage patterns");
    let fake_usageviolations = scan_for_fake_usage();
    let fake_usage_report = format!(
        "fake usage scan identified {} violation groups",
        fake_usageviolations.len()
    );
    emit_stage_detail(&fake_usage_report);
    allviolations.extend(fake_usageviolations);
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

    // If any violations were found, print them all and exit with error
    if !allviolations.is_empty() {
        update_stage("report validation errors");
        eprintln!("\n❌ VALIDATION ERRORS");
        eprintln!("====================");

        let violation_count = allviolations.len();

        for violation in allviolations {
            eprintln!("{violation}");
            eprintln!("--------------------");
        }

        eprintln!(
            "\n⚠️ Found {} total code quality violations. Fix all issues before committing.",
            violation_count
        );
        std::process::exit(1);
    }

    update_stage("build script completed");
    emit_stage_detail("Validation checks completed without errors");
}

// This function manually checks for unused variables in the current file
fn manually_check_for_unusedvariables() {
    // Force compilation to fail with unusedvariables, dead_code, and unused_imports lint
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
                    std::process::exit(1);
                } else if stderr.contains("function is never used") {
                    eprintln!("\n❌ ERROR: Unused functions detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused functions are STRICTLY FORBIDDEN in this project.");
                    eprintln!("   Either use the function or remove it completely.");
                    std::process::exit(1);
                } else if stderr.contains("unused import") {
                    eprintln!("\n❌ ERROR: Unused imports detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused imports are STRICTLY FORBIDDEN in this project.");
                    eprintln!("   Either use the imported item or remove the import completely.");
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
        OsString::from("unusedvariables"),
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

                // Check if we can read the file
                match std::fs::read_to_string(path) {
                    Ok(_) => {}         // File exists and can be read
                    Err(_) => continue, // Skip files we can't read
                };

                // Add debug info for estimate.rs to help diagnose the underscore variable detection
                let is_estimate_rs = path
                    .to_str()
                    .is_some_and(|p| p.ends_with("calibrate/estimate.rs"));
                if is_estimate_rs && warnings_enabled() {
                    println!(
                        "cargo:warning=Analyzing estimate.rs for underscore-prefixed variables"
                    );
                }

                // Create a new collector for each file.
                let mut collector = ViolationCollector::new(path);

                // Search the file using our regex matcher and collector sink.
                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = DisallowedLetCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = NoopTouchCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = TupleWildcardCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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
    // Regex patterns to find forbidden comment patterns
    // Note: We specifically target comments by looking for // or /* */ patterns
    // This ensures we don't flag these terms in actual code
    let mut allviolations = Vec::new();

    // Split into separate patterns for clarity and reliability
    // 1. Pattern to catch forbidden words in comments
    let forbiddenwords_pattern = r"(//|/\*|///).*(?:DEPRECATED|CRITICAL|FIXED|CORRECTED|FIX|FIXES|NEW|CHANGED|CHANGES|CHANGE|MODIFIED|MODIFIES|MODIFY|UPDATED|UPDATES|UPDATE)";
    // 2. Pattern to catch ** in comments (excluding doc comments)
    let stars_pattern = r"(//|/\*).*\*\*";
    // 3. Pattern to catch comments for uppercase ratio enforcement
    let all_caps_pattern = r"(//|/\*|///).*";
    // 4. Pattern to catch comments that might be composed primarily of dashes
    let dash_heavy_pattern = r"(//|/\*|///).*";

    // Check for forbidden words
    match RegexMatcher::new_line_matcher(forbiddenwords_pattern) {
        Ok(forbidden_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path())) // Exclude ignored directories
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                // Use a collector that doesn't filter out doc comments for forbidden words
                let mut collector = ForbiddenCommentCollector::new(path, true);
                if searcher
                    .search_path(&forbidden_matcher, path, &mut collector)
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
            // Record the error but continue checking other patterns
            allviolations.push(format!("Error creating forbidden words regex: {}", e));
        }
    }

    // Check for stars in non-doc comments
    match RegexMatcher::new_line_matcher(stars_pattern) {
        Ok(stars_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path())) // Exclude ignored directories
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                // Use a single collector with custom filtering logic
                // false means don't check for ** in doc comments
                let mut collector = ForbiddenCommentCollector::new(path, false);
                if searcher
                    .search_path(&stars_matcher, path, &mut collector)
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
            // Record the error but continue checking other patterns
            allviolations.push(format!("Error creating stars pattern regex: {}", e));
        }
    }

    // Check for comments where the uppercase ratio exceeds the threshold
    match RegexMatcher::new_line_matcher(all_caps_pattern) {
        Ok(all_caps_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let mut custom_collector = CustomUppercaseCollector::new(path);
                if searcher
                    .search_path(&all_caps_matcher, path, &mut custom_collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = custom_collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            // Record the error but don't return early
            allviolations.push(format!("Error creating uppercase pattern regex: {}", e));
        }
    }

    // Check for comments composed primarily of dashes
    match RegexMatcher::new_line_matcher(dash_heavy_pattern) {
        Ok(dash_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
                .filter(|e: &walkdir::DirEntry| !is_in_ignored_directory(e.path()))
                .filter(|e: &walkdir::DirEntry| e.file_name() != "build.rs")
                .filter(|e: &walkdir::DirEntry| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let mut dash_collector = DashHeavyCommentCollector::new(path);
                if searcher
                    .search_path(&dash_matcher, path, &mut dash_collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = dash_collector.check_and_get_error_message() {
                    allviolations.push(error_message);
                }
            }
        }
        Err(e) => {
            allviolations.push(format!("Error creating dash-heavy pattern regex: {}", e));
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
    // - #[allow(unusedvariables)]
    // - #[allow(..., unused, ...)] (inside lists)
    let pattern = r"#!?\s*\[\s*allow\s*\([^)]*\b(dead_code|unused|unused_imports|unusedvariables)\b[^)]*\)\s*\]";
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

                // Check if we can read the file
                match std::fs::read_to_string(path) {
                    Ok(_) => {}         // File exists and can be read
                    Err(_) => continue, // Skip files we can't read
                };

                // Create a collector for each file
                let mut collector = DeadCodeCollector::new(path);

                // Search the file using our regex matcher and collector sink
                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                // Check if we can read the file
                match std::fs::read_to_string(path) {
                    Ok(_) => {}         // File exists and can be read
                    Err(_) => continue, // Skip files we can't read
                };

                // Create a collector for each file
                let mut collector = IgnoredTestCollector::new(path);

                // Search the file using our regex matcher and collector sink
                if searcher
                    .search_path(&matcher, path, &mut collector)
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
    // Regex pattern to find TODO/FIXME/"for now" comments (case insensitive)
    // Matches: TODO, FIXME, XXX, "for now" in comments
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

                match std::fs::read_to_string(path) {
                    Ok(_) => {}
                    Err(_) => continue,
                };

                let mut collector = TodoCommentCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = DropUsageCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = DropUsageCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = DebugAssertCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = MeaninglessConditionalCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = NoEffectCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = OmittedForBrevityCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                if std::fs::read_to_string(path).is_err() {
                    continue;
                }

                let mut collector = DeprecatedCollector::new(path);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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

                // Find all Ok(Self { matches
                let mut collector = PlaceholderStubCollector::new(path, &content);

                if searcher
                    .search_path(&matcher, path, &mut collector)
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
                    out.push(b'x');
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
                allviolations.push(msg);
            }
        }
    }
    allviolations
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

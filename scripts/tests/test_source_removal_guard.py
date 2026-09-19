import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("source_removal_guard", Path(__file__).parents[1] / "source_removal_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def commit(root, files):
    """Write `files` (None deletes) into the repository at `root` and commit them."""
    for path, text in files.items():
        target = root / path
        if text is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=guard", "-c", "user.email=guard@example.invalid",
                    "commit", "-q", "-m", "revision"], check=True)
    return guard.resolve(root, "HEAD")


class SourceRemovalGuard(unittest.TestCase):
    def test_test_scoped_helper_cannot_be_declared_unreachable_2818(self):
        source = """
        fn live() { helper(); }
        #[cfg(test)]
        mod tests {
            pub(crate) fn helper() {}
        }
        """
        marked = guard.test_lines(source, "crates/demo/src/lib.rs")
        self.assertIn(5, marked)
        before = {"p:fn:helper": {"path": "p", "line": 5, "kind": "fn", "name": "helper",
                                   "public": True, "test_scoped": True}}
        self.assertEqual(guard.removals(before, {}, {"helper": 0})[0]["name"], "helper")

    def test_public_generic_without_a_linked_symbol_is_guarded_2818(self):
        item = {"path": "src/lib.rs", "line": 1, "kind": "fn", "name": "map_value",
                "public": True, "test_scoped": False}
        self.assertEqual(len(guard.removals({"x": item}, {}, {"map_value": 1})), 1)

    def test_private_item_needs_source_graph_not_symbol_table_2818(self):
        item = {"path": "src/lib.rs", "line": 1, "kind": "fn", "name": "folded",
                "public": False, "test_scoped": False}
        self.assertEqual(len(guard.removals({"x": item}, {}, {"folded": 2})), 1)
        self.assertEqual(guard.removals({"x": item}, {}, {"folded": 1}), [])

    def test_guarded_removal_requires_exact_semantic_record_2818(self):
        blocked = [{"path": "src/lib.rs", "kind": "fn", "name": "api"}]
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            guard.check("base", blocked, [])
        entry = {"base": "base", "items": ["src/lib.rs:fn:api"],
                 "reason": "The behavior was retired.", "evidence": "Reviewed in #1234."}
        guard.check("base", blocked, [entry])

    def test_lifetimes_and_raw_strings_cannot_hide_declarations_2818(self):
        source = ("pub fn borrow<'a>(x: &'a str) -> &'a str { x }\n"
                  "pub static LABEL: &'static str = \"a 'quoted' label\";\n"
                  "const QUOTE: char = '\\'';\n"
                  "const RAW: &str = r#\"pub fn fake() {} \" still inside\"#;\n"
                  "/* outer /* pub fn nested_fake() {} */ still comment */\n"
                  "pub fn after() {}\n")
        clean = guard.strip_comments_and_literals(source)
        self.assertEqual(clean.count("\n"), source.count("\n"), "line numbers must survive stripping")
        names = [match["name"] for match in map(guard.ITEM.match, clean.splitlines()) if match]
        self.assertEqual(names, ["borrow", "LABEL", "QUOTE", "RAW", "after"])

    def test_an_out_of_line_test_module_file_is_test_scope_2818(self):
        source = "pub(crate) fn fixture() {}\nfn helper() {}\n"
        for path in ("crates/demo/src/manifold/tests_logdet_adjoint_780.rs",
                     "crates/demo/src/solver_tests.rs", "crates/demo/src/test_support.rs"):
            with self.subTest(path=path):
                self.assertEqual(guard.test_lines(source, path), {1, 2, 3})
        self.assertEqual(guard.test_lines(source, "crates/demo/src/testsuite.rs"), set())

    def test_a_brace_less_test_item_ends_at_its_semicolon_2829(self):
        source = ("#[cfg(test)]\n"
                  "mod tests_x;\n"
                  "pub fn production() {\n"
                  "}\n"
                  "#[cfg(test)] use std::fmt;\n"
                  "#[cfg(test)]\n"
                  "const TABLE: [u8; 2] = [1, 2];\n"
                  "fn also_production() {}\n"
                  "#[cfg(test)]\n"
                  "mod tests {\n"
                  "    fn helper() {}\n"
                  "}\n")
        self.assertEqual(guard.test_lines(source, "crates/demo/src/lib.rs"), {1, 2, 5, 6, 7, 9, 10, 11, 12})

    def test_a_line_that_merely_starts_with_a_keyword_is_not_a_declaration_2818(self):
        for line in ("    constant_curvature_kernel_matrix(x, y)", "enumerate_generators(&mut out);",
                     "    type_per_point_log_density = 3;", "fnord();", "static_bound.check()"):
            with self.subTest(line=line):
                self.assertIsNone(guard.ITEM.match(line))

    def test_qualified_declarations_keep_their_own_names_2818(self):
        for line, identity in (("pub const fn width() -> usize {", ("fn", "width")),
                               ("pub(crate) async unsafe fn go() {", ("fn", "go")),
                               ('unsafe extern "C" fn callback() {', ("fn", "callback")),
                               ("pub const MAX: usize = 3;", ("const", "MAX")),
                               ("static mut COUNTER: u8 = 0;", ("static", "COUNTER")),
                               ("macro_rules! probe {", ("macro_rules!", "probe"))):
            with self.subTest(line=line):
                match = guard.ITEM.match(line)
                self.assertEqual((match["kind"], match["name"]), identity)

    def test_losing_one_of_several_same_named_declarations_is_a_removal_2818(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            # Declarations sit on their own lines, as rustfmt writes them; the
            # inventory reads items at line starts.
            base = commit(root, {"src/lib.rs": "pub struct A;\npub struct B;\n"
                                              "impl A {\n    pub fn new() -> Self {\n        A\n    }\n}\n"
                                              "impl B {\n    pub fn new() -> Self {\n        B\n    }\n}\n"
                                              "pub const fn width() -> usize {\n    1\n}\n"
                                              "pub const fn height() -> usize {\n    2\n}\n"})
            head = commit(root, {"src/lib.rs": "pub struct A;\npub struct B;\n"
                                              "impl A {\n    pub fn new() -> Self {\n        A\n    }\n}\n"
                                              "pub const fn width() -> usize {\n    1\n}\n"})
            before, after, words = guard.changed_inventory(root, base, head)
            blocked = {(x["name"], x["removed_declarations"]) for x in guard.removals(before, after, words)}
            self.assertEqual(blocked, {("new", 1), ("height", 1)})

    def test_a_public_homonym_cannot_launder_a_private_items_callers_2818(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            # `crates/a` is inventoried first, so the public homonym in `crates/b`
            # is seen after the private item whose callers decide its removal.
            base = commit(root, {"crates/a/src/lib.rs": "fn shared() {}\nfn kept() {}\nfn user() { shared(); kept(); }\n",
                                 "crates/b/src/lib.rs": "pub fn shared() {}\n"})
            head = commit(root, {"crates/a/src/lib.rs": "fn kept() {}\nfn user() { shared(); kept(); }\n",
                                 "crates/b/src/lib.rs": "\n"})
            with mock.patch.object(guard, "source_words", wraps=guard.source_words) as spy:
                before, after, words = guard.changed_inventory(root, base, head)
            self.assertEqual(spy.call_count, 1, "one pass counts every removed private name")
            self.assertEqual(spy.call_args.args[1:], (base, ["shared"]),
                             "only the private name that lost a declaration is queried")
            blocked = {f"{x['path']}:{x['kind']}:{x['name']}" for x in guard.removals(before, after, words)}
            self.assertEqual(blocked, {"crates/a/src/lib.rs:fn:shared", "crates/b/src/lib.rs:fn:shared"})

    def test_a_private_item_nobody_calls_is_removable_2818(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            base = commit(root, {"src/lib.rs": "fn orphan() {}\npub fn live() {}\n"})
            head = commit(root, {"src/lib.rs": "pub fn live() {}\n"})
            before, after, words = guard.changed_inventory(root, base, head)
            self.assertEqual(words["orphan"], 1, "the declaration itself is the only occurrence")
            self.assertEqual(guard.removals(before, after, words), [])

    def test_a_renamed_file_is_compared_with_its_new_path_2818(self):
        # c7768c15c2 renamed empirical_intercept_bracket_tests.rs and dropped two of
        # its tests. `diff --name-only` listed only the new path, so the old one was
        # never read and the guard passed it. diff.renames=false here proves the
        # pairing does not come from configuration.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            subprocess.run(["git", "-C", directory, "config", "diff.renames", "false"], check=True)
            kept = "".join(f"#[test]\nfn kept_{i}() {{\n    assert_eq!({i}, {i});\n}}\n" for i in range(12))
            base = commit(root, {"crates/demo/src/bracket_tests.rs": kept + "#[test]\nfn dropped() {\n    assert_eq!(12, 12);\n}\n"})
            renamed = commit(root, {"crates/demo/src/bracket_tests.rs": None, "crates/demo/src/solve_tests.rs": kept})
            before, after, words = guard.changed_inventory(root, base, renamed)
            blocked = [f"{x['path']}:{x['kind']}:{x['name']}" for x in guard.removals(before, after, words)]
            self.assertEqual(blocked, ["crates/demo/src/bracket_tests.rs:fn:dropped"])
            moved = commit(root, {"crates/demo/src/solve_tests.rs": None, "crates/other/src/solve_tests.rs": kept})
            before, after, words = guard.changed_inventory(root, renamed, moved)
            self.assertEqual(len(before), 12, "the old path is read")
            self.assertEqual(guard.removals(before, after, words), [], "a pure move removes nothing")

    def test_a_pub_fn_deleted_inside_a_moved_file_is_refused_2818(self):
        # Git's default rename detection is on here, the configuration under which
        # `diff --name-only` hid the old path.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            kept = "".join(f"pub fn kept_{i}() -> usize {{\n    {i}\n}}\n" for i in range(12))
            base = commit(root, {"crates/demo/src/model.rs": kept + "pub fn dropped() -> usize {\n    12\n}\n"})
            head = commit(root, {"crates/demo/src/model.rs": None, "crates/other/src/model.rs": kept})
            before, after, words = guard.changed_inventory(root, base, head)
            blocked = [f"{x['path']}:{x['kind']}:{x['name']}" for x in guard.removals(before, after, words)]
            self.assertEqual(blocked, ["crates/demo/src/model.rs:fn:dropped"])

    def test_inline_format_captures_are_uses_but_escaped_braces_are_text_2818(self):
        source = ('pub const REFERENCE_ENV_MISSING: &str = "REFERENCE_ENV_MISSING";\n'
                  'fn report(tool: &str) -> String {\n'
                  '    format!("{REFERENCE_ENV_MISSING}:{tool}: not installed, see {{literal}}")\n'
                  '}\n'
                  'fn padded(width: usize) -> String {\n'
                  '    format!(r#"{:>width$} {{literal}}"#, 1)\n'
                  '}\n')
        clean = guard.strip_comments_and_literals(source)
        self.assertEqual(clean.count("\n"), source.count("\n"), "line numbers must survive stripping")
        lines = clean.splitlines()
        self.assertEqual(re.findall(r"\w+", lines[0]).count("REFERENCE_ENV_MISSING"), 1,
                         "a plain string holding the name is text, not a use")
        self.assertIn("REFERENCE_ENV_MISSING", re.findall(r"\w+", lines[2]), "a `{name}` capture is a use")
        self.assertIn("tool", re.findall(r"\w+", lines[2]))
        self.assertIn("width", re.findall(r"\w+", lines[5]), "a `name$` width inside a spec is a use")
        self.assertNotIn("literal", clean, "an escaped `{{...}}` is text, not a capture")
        self.assertNotIn("installed", clean, "the rest of the literal stays blank")


if __name__ == "__main__":
    unittest.main()

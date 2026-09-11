import importlib.util
from pathlib import Path
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
            with mock.patch.object(guard.subprocess, "run", wraps=subprocess.run) as spy:
                before, after, words = guard.changed_inventory(root, base, head)
            queries = [call.args[0] for call in spy.call_args_list if "grep" in call.args[0]]
            self.assertEqual(len(queries), 1, "one pass counts every removed private name")
            self.assertEqual(queries[0][queries[0].index("-F") + 1:], ["-e", "shared", base, "--", "*.rs"],
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


if __name__ == "__main__":
    unittest.main()

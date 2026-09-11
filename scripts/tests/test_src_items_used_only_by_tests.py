import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
SPEC = importlib.util.spec_from_file_location("src_items_used_only_by_tests",
                                              Path(__file__).parents[1] / "src_items_used_only_by_tests.py")
scanner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scanner)

EMPTY = {"test_only": [], "unreferenced": []}


def identities(report, kind):
    return sorted(entry["identity"] for entry in report[kind])


class SrcItemsUsedOnlyByTests(unittest.TestCase):
    def test_an_item_only_a_test_names_is_test_only_2818(self):
        report = scanner.scan({
            "crates/a/src/lib.rs": "pub(crate) fn helper() -> u8 { 3 }\n"
                                   "#[cfg(test)]\nmod tests {\n    #[test]\n"
                                   "    fn uses() { assert_eq!(super::helper(), 3); }\n}\n",
        })
        self.assertEqual(identities(report, "test_only"), ["crates/a/src/lib.rs:fn:helper"])
        self.assertEqual(report["test_only"][0]["test_reference"], "crates/a/src/lib.rs")
        self.assertEqual(report["unreferenced"], [])

    def test_a_production_caller_clears_it_in_another_file_or_its_own_2818(self):
        across = scanner.scan({
            "crates/a/src/lib.rs": "pub(crate) fn helper() -> u8 { 3 }\n",
            "crates/a/src/user.rs": "pub fn live() -> u8 { crate::helper() }\n",
            "crates/a/tests/suite.rs": "#[test] fn t() { assert_eq!(a::live(), 3); }\n",
        })
        self.assertEqual(across, EMPTY)
        within = scanner.scan({"crates/a/src/lib.rs": "pub(crate) fn helper() {}\npub fn live() { helper() }\n"})
        self.assertEqual(within, EMPTY)

    def test_comments_and_literals_cannot_supply_a_production_caller_2818(self):
        report = scanner.scan({
            "crates/a/src/lib.rs": "pub(crate) fn helper() {}\n",
            "crates/a/src/other.rs": "// helper()\nconst NAME: &str = \"helper\";\n/* helper */\n"
                                     "const RAW: &str = r#\"helper\"#;\n",
            "crates/a/src/tests_helper.rs": "#[test] fn t() { crate::helper(); }\n",
        })
        self.assertEqual(identities(report, "test_only"), ["crates/a/src/lib.rs:fn:helper"])

    def test_bare_pub_trait_members_short_and_exempt_names_are_out_of_scope_2818(self):
        report = scanner.scan({
            "crates/a/src/lib.rs": "pub fn api() {}\n"
                                   "pub(crate) struct Thing;\n"
                                   "impl Default for Thing {\n    fn default() -> Self { Thing }\n}\n"
                                   "impl std::fmt::Display for Thing {\n"
                                   "    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { Ok(()) }\n"
                                   "    fn render_label(&self) {}\n}\n"
                                   "pub(crate) fn ab() {}\n",
        })
        self.assertEqual(report, EMPTY)

    def test_trait_definition_members_are_dispatched_but_a_free_function_is_still_scanned_2818(self):
        # Both shapes were reported on main at c0b51f0a4 before trait bodies were masked:
        # a required trait method and a default method, each named only by a test.
        report = scanner.scan({
            "crates/a/src/lib.rs": "pub trait Objective {\n"
                                   "    fn value_and_gradient(&mut self) -> f64;\n"
                                   "    fn eigendecompose(&self) -> f64 {\n        1.0\n    }\n}\n"
                                   "pub(crate) fn free_helper() {}\n",
            "crates/a/tests/suite.rs": "#[test]\nfn t() {\n    x.value_and_gradient();\n"
                                       "    x.eigendecompose();\n    a::free_helper();\n}\n",
        })
        self.assertEqual(identities(report, "test_only"), ["crates/a/src/lib.rs:fn:free_helper"])
        self.assertEqual(report["unreferenced"], [])

    def test_an_unreferenced_crate_item_is_reported_and_a_private_one_left_to_rustc_2818(self):
        report = scanner.scan({"crates/a/src/lib.rs": "pub(crate) fn orphan() {}\nfn private_orphan() {}\n"})
        self.assertEqual(identities(report, "unreferenced"), ["crates/a/src/lib.rs:fn:orphan"])
        self.assertEqual(report["test_only"], [])

    def test_a_lifetime_cannot_hide_the_declarations_after_it_2818(self):
        report = scanner.scan({
            "crates/a/src/lib.rs": "pub fn borrow<'a>(x: &'a str) -> &'a str { x }\n"
                                   "pub(crate) static LABEL: &'static str = \"x\";\n"
                                   "pub(crate) const QUOTE: char = '\\'';\n"
                                   "pub(crate) fn later() {}\n",
        })
        self.assertEqual(identities(report, "unreferenced"), ["crates/a/src/lib.rs:const:QUOTE",
                                                               "crates/a/src/lib.rs:fn:later",
                                                               "crates/a/src/lib.rs:static:LABEL"])

    def test_a_definition_inside_test_scope_is_never_a_candidate_2818(self):
        report = scanner.scan({
            "crates/a/src/lib.rs": "#[cfg(test)]\nmod test_support {\n    pub(crate) fn fixture() {}\n}\n",
            "crates/a/src/tests_fixture.rs": "pub(crate) fn other_fixture() {}\n",
            "crates/a/src/solver_tests.rs": "pub(crate) fn third_fixture() {}\n",
        })
        self.assertEqual(report, EMPTY)

    def test_a_crate_root_reexport_is_a_production_consumer_2818(self):
        report = scanner.scan({
            "crates/a/src/lib.rs": "mod inner;\npub use inner::{Exported, other as Renamed};\n",
            "crates/a/src/inner.rs": "pub(crate) struct Exported;\npub(crate) fn Renamed() {}\n",
        })
        self.assertEqual(report, EMPTY)

    def test_the_ledger_ratchets_in_both_directions_2818(self):
        report = {"test_only": [{"identity": "crates/a/src/lib.rs:fn:helper", "line": 1, "test_reference": "x"}],
                  "unreferenced": [{"identity": "crates/a/src/lib.rs:fn:new_orphan", "line": 2}]}
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.txt"
            ledger.write_text("# known findings\ntest-only crates/a/src/lib.rs:fn:helper\n"
                              "unreferenced crates/a/src/lib.rs:fn:gone\n")
            regressions, stale = scanner.ratchet(report, scanner.read_ledger(ledger))
            self.assertEqual(regressions, ["unreferenced crates/a/src/lib.rs:fn:new_orphan"])
            self.assertEqual(stale, ["unreferenced crates/a/src/lib.rs:fn:gone"])
            ledger.write_text("unreferenced b\nunreferenced a\n")
            with self.assertRaisesRegex(ValueError, "not sorted"):
                scanner.read_ledger(ledger)


if __name__ == "__main__":
    unittest.main()

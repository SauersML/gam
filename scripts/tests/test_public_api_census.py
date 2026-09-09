import importlib.util
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
SPEC = importlib.util.spec_from_file_location(
    "public_api_census", Path(__file__).parents[1] / "public_api_census.py"
)
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


class PublicApiCensus(unittest.TestCase):
    def test_generic_public_function_is_a_source_root_even_without_a_binary_caller_2829(self):
        self.assertEqual(list(census.public_functions("pub fn fit_gam<X>(x: X) {}")), ["fit_gam"])

    def test_comments_and_literals_cannot_forge_a_surviving_definition_2829(self):
        source = '''
        // pub fn fit_gam<X>() {}
        const ADVICE: &str = "use fit_gam/predict_gam";
        /* pub(crate) fn fit_gam() {} */
        pub async unsafe fn actual() {}
        '''
        self.assertEqual(list(census.public_functions(source)), ["actual"])

    def test_restricted_visibility_qualifiers_and_raw_identifiers_are_counted_2829(self):
        source = '''
        pub(crate) const unsafe extern "C" fn crate_api() {}
        pub(super) fn parent_api() {}
        pub(in crate::module) async fn r#move() {}
        fn private() {}
        '''
        self.assertEqual(list(census.public_functions(source)),
                         ["crate_api", "parent_api", "move"])

    def test_macro_metavariable_is_not_mistaken_for_a_source_declaration_2829(self):
        self.assertEqual(list(census.public_functions("macro_rules! m { ($n:ident) => { pub fn $n() {} } }")), [])

    def test_same_name_in_another_file_does_not_mask_a_removal_2829(self):
        before = snapshot({"crates/a/src/lib.rs::under_identified_subspace": 1})
        after = snapshot({"crates/b/src/lib.rs::under_identified_subspace": 1})
        self.assertEqual(census.difference(before, after),
                         {"crates/a/src/lib.rs::under_identified_subspace": 1})
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            census.check_change(before, after, [])

    def test_removal_requires_exact_semantic_acknowledgement_2829(self):
        before = snapshot({"crates/a/src/lib.rs::fit_gam": 1})
        after = snapshot({})
        entry = {"base": "base", "removed": {"crates/a/src/lib.rs::fit_gam": 1},
                 "reason": "The explicit-design entry supersedes the defaulted alias.",
                 "evidence": "external consumer replacement test: 1 passed"}
        self.assertEqual(census.check_change(before, after, [entry]), entry["removed"])
        for key, value in (("reason", ""), ("evidence", ""), ("removed", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                census.check_change(before, after, [dict(entry, **{key: value})])


def snapshot(identities):
    return {"revision": "base", "identities": Counter(identities)}


if __name__ == "__main__":
    unittest.main()

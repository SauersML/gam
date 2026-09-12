import importlib.util
from pathlib import Path
import unittest
from collections import Counter


SPEC = importlib.util.spec_from_file_location("test_census", Path(__file__).parents[1] / "test_census.py")
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


class TestCensus(unittest.TestCase):
    def test_deleting_all_tests_and_annotations_cannot_pass_2818(self):
        for files, names, pins in ((0, Counter(), Counter()), (30, Counter(), Counter()),
                                   (30, Counter({"ordinary": 2}), Counter())):
            with self.subTest(files=files, names=names), self.assertRaisesRegex(ValueError, "empty census denominator"):
                census.validate_population("head", files, names, pins)

    def test_comments_literals_and_lifetimes_cannot_supply_pins_2818(self):
        source = '''
        // #[test] fn vanished_12() {}
        /* outer /* #[test] fn hidden_13() {} */ comment */
        const DOC: &str = r###" #[test] fn fake_14() {} "###;
        const BYTE: &[u8] = br##" #[test] fn fake_15() {} "##;
        const QUOTE: char = '\\'';
        fn borrow<'a>(x: &'a str) -> &'a str { x }
        #[test] #[should_panic(expected = "fn wrong_16")]
        fn actual_17() { let s = "#[test] fn fake_18() {}"; }
        #[cfg_attr(feature = "oracle", test)] fn conditional_19() {}
        '''
        self.assertEqual(list(census.test_names(source)), ["actual_17", "conditional_19"])

    def test_annotated_missing_function_is_a_failure_2818(self):
        for source in ("#[test]", "#[test] mod empty {}", "#[test] #[test] fn x() {}"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                list(census.test_names(source))

    def test_removing_a_pin_is_reported_when_total_count_grows_2818(self):
        before = snapshot(3, {"critical_2818": 1}, {"crates/gam-sae": 3})
        after = snapshot(4, {"unrelated_2817": 3}, {"crates/gam-sae": 4})
        self.assertEqual(census.difference(before, after), {
            "test_count_decrease": 0, "removed_pins": {"critical_2818": 1}, "unit_test_decreases": {}})

    def test_duplicate_pin_loss_and_plain_test_count_loss_are_visible_2818(self):
        before = snapshot(5, {"critical_12": 2})
        after = snapshot(4, {"critical_12": 1})
        self.assertEqual(census.difference(before, after), {
            "test_count_decrease": 1, "removed_pins": {"critical_12": 1}, "unit_test_decreases": {}})

    def test_total_count_loss_is_reported_even_when_every_pin_survives_2818(self):
        before = snapshot(5, {"critical_2818": 1}, {"crates/gam-sae": 5})
        after = snapshot(4, {"critical_2818": 1}, {"crates/gam-sae": 4})
        self.assertEqual(census.difference(before, after), {
            "test_count_decrease": 1, "removed_pins": {}, "unit_test_decreases": {"crates/gam-sae": 1}})

    def test_growth_with_unchanged_identities_reports_no_loss_2818(self):
        before = snapshot(5, {"critical_12": 1})
        after = snapshot(6, {"critical_12": 1, "new_13": 1})
        self.assertEqual(census.difference(before, after), {
            "test_count_decrease": 0, "removed_pins": {}, "unit_test_decreases": {}})

    def test_growth_in_one_crate_cannot_pay_for_deletion_in_another_2818(self):
        """The workspace total is blind to the shape #2818 actually had."""
        before = snapshot(40, {"critical_12": 1}, {"crates/gam-sae": 30, "crates/gam-solve": 10})
        after = snapshot(40, {"critical_12": 1}, {"crates/gam-sae": 5, "crates/gam-solve": 35})
        self.assertEqual(census.difference(before, after)["test_count_decrease"], 0)
        self.assertEqual(census.difference(before, after)["unit_test_decreases"], {"crates/gam-sae": 25})

    def test_units_group_by_crate_and_by_top_level_suite_2818(self):
        self.assertEqual(census.unit("crates/gam-sae/src/manifold/mod.rs"), "crates/gam-sae")
        self.assertEqual(census.unit("crates/gam-sae/tests/atlas.rs"), "crates/gam-sae")
        self.assertEqual(census.unit("src/lib.rs"), "src")

    def test_the_tests_tree_is_split_by_integration_binary_2818(self):
        """`tests` was ONE unit of 2,619 — larger than any crate. A unit that is
        itself an aggregate nets inside itself, which is the same defect the
        per-crate split fixed one level up."""
        self.assertEqual(census.unit("tests/regressions/predict/dispersion.rs"), "tests/regressions")
        self.assertEqual(census.unit("tests/sae/main.rs"), "tests/sae")
        self.assertEqual(census.unit("tests/common/mod.rs"), "tests/common")
        # Both spellings of one binary are one unit, or moving a suite from a
        # single file into a directory would read as a whole unit disappearing.
        self.assertEqual(census.unit("tests/measure_jet_ctn_range_screen_2754.rs"),
                         "tests/measure_jet_ctn_range_screen_2754")
        self.assertEqual(census.unit("tests/measure_jet_ctn_range_screen_2754/main.rs"),
                         "tests/measure_jet_ctn_range_screen_2754")

    def test_a_deletion_inside_tests_is_no_longer_paid_for_by_growth_beside_it_2818(self):
        """The netting the split exists to kill, as a before/after on one input.

        Three tests leave `tests/regressions` and three arrive in `tests/sae`.
        Under one lumped `tests` unit the loss is invisible; under the split it
        is named.
        """
        before = snapshot(632, {"pinned_2818": 1}, {"tests/regressions": 506, "tests/sae": 126})
        after = snapshot(632, {"pinned_2818": 1}, {"tests/regressions": 503, "tests/sae": 129})
        self.assertEqual(census.difference(before, after)["unit_test_decreases"], {"tests/regressions": 3})

        lumped_before = snapshot(632, {"pinned_2818": 1}, {"tests": 506 + 126})
        lumped_after = snapshot(632, {"pinned_2818": 1}, {"tests": 503 + 129})
        self.assertEqual(census.difference(lumped_before, lumped_after)["unit_test_decreases"], {})


def snapshot(tests, pins, units=None):
    return {"revision": "base", "tests": tests, "pins": Counter(pins), "units": Counter(units or {})}


if __name__ == "__main__":
    unittest.main()

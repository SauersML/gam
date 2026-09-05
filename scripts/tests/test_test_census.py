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

    def test_removing_a_pin_still_fails_when_total_count_grows_2818(self):
        before = snapshot(3, {"critical_12": 1})
        after = snapshot(4, {"unrelated_13": 3})
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            census.check_change(before, after, [])

    def test_duplicate_pin_loss_and_plain_test_count_loss_are_visible_2818(self):
        before = snapshot(5, {"critical_12": 2})
        after = snapshot(4, {"critical_12": 1})
        self.assertEqual(census.difference(before, after), {
            "test_count_decrease": 1, "removed_pins": {"critical_12": 1}, "unit_test_decreases": {}})
        with self.assertRaises(ValueError):
            census.check_change(before, after, [])

    def test_total_count_loss_fails_even_when_every_pin_survives_2818(self):
        before = snapshot(5, {"critical_12": 1})
        after = snapshot(4, {"critical_12": 1})
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            census.check_change(before, after, [])

    def test_unchanged_identities_need_no_removal_acknowledgement_2818(self):
        before = snapshot(5, {"critical_12": 1})
        after = snapshot(6, {"critical_12": 1, "new_13": 1})
        self.assertEqual(census.check_change(before, after, []), {
            "test_count_decrease": 0, "removed_pins": {}, "unit_test_decreases": {}})

    def test_acknowledgement_requires_exact_loss_and_semantic_evidence_2818(self):
        before = snapshot(2, {"critical_12": 1})
        after = snapshot(1, {})
        entry = {"base": "base", "test_count_decrease": 1, "unit_test_decreases": {},
                 "removed_pins": {"critical_12": 1}, "reason": "Replacement covers the same derivative",
                 "evidence": "replacement_12 executed: 1 passed"}
        self.assertEqual(census.check_change(before, after, [entry]), census.difference(before, after))
        for field, value in (("base", "stale"), ("test_count_decrease", 2), ("evidence", ""), ("removed_pins", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                census.check_change(before, after, [dict(entry, **{field: value})])

    def test_growth_in_one_crate_cannot_pay_for_deletion_in_another_2818(self):
        """The workspace total is blind to the shape #2818 actually had."""
        before = snapshot(40, {"critical_12": 1}, {"crates/gam-sae": 30, "crates/gam-solve": 10})
        after = snapshot(40, {"critical_12": 1}, {"crates/gam-sae": 5, "crates/gam-solve": 35})
        self.assertEqual(census.difference(before, after)["test_count_decrease"], 0)
        self.assertEqual(census.difference(before, after)["unit_test_decreases"], {"crates/gam-sae": 25})
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            census.check_change(before, after, [])

    def test_floor_holds_units_and_issues_without_a_base_commit_2818(self):
        """The floor is the assertion that survives a broken incremental chain."""
        floor = {"generated_from": "mark", "units": {"crates/gam-sae": 30}, "issues": {"2818": 4}}
        self.assertEqual(census.check_floor(snapshot(40, {}, {"crates/gam-sae": 31}, {"2818": 9}), floor), {})
        for units, issues in (({"crates/gam-sae": 29}, {"2818": 4}), ({"crates/gam-sae": 30}, {"2818": 3}),
                              ({}, {}), ({"crates/gam-solve": 99}, {"2817": 99})):
            with self.subTest(units=units), self.assertRaisesRegex(ValueError, "below the recorded floor"):
                census.check_floor(snapshot(40, {}, units, issues), floor)

    def test_floor_rejects_a_control_file_that_is_not_counts_2818(self):
        good = {"generated_from": "mark", "units": {"crates/gam-sae": 1}, "issues": {"2818": 1}}
        measured = snapshot(40, {}, {"crates/gam-sae": 9}, {"2818": 9})
        self.assertEqual(census.check_floor(measured, good), {})
        for broken in ({"units": {}, "issues": {}}, dict(good, extra=1),
                       dict(good, units={"crates/gam-sae": "1"}), dict(good, issues={"2818": -1}),
                       dict(good, units={"crates/gam-sae": True})):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                census.check_floor(measured, broken)

    def test_a_renamed_pin_keeps_answering_for_its_issue_2818(self):
        """Names churn; the bug a test pins does not. The floor groups by issue."""
        self.assertEqual(census.issue_numbers("co_routed_frame_sweep_is_tied_code_descent_2634"), ("2634",))
        self.assertEqual(census.issue_numbers("end_to_end_parity_battery_2156_2144"), ("2156", "2144"))
        self.assertEqual(census.issue_numbers("plain_name"), ())
        self.assertEqual(census.issue_numbers("matrix_is_2x2_12"), ())

    def test_units_group_by_crate_and_by_top_level_suite_2818(self):
        self.assertEqual(census.unit("crates/gam-sae/src/manifold/mod.rs"), "crates/gam-sae")
        self.assertEqual(census.unit("crates/gam-sae/tests/atlas.rs"), "crates/gam-sae")
        self.assertEqual(census.unit("tests/gam_sae.rs"), "tests")
        self.assertEqual(census.unit("src/lib.rs"), "src")


def snapshot(tests, pins, units=None, issues=None):
    return {"revision": "base", "tests": tests, "pins": Counter(pins),
            "units": Counter(units or {}), "issues": Counter(issues or {})}


if __name__ == "__main__":
    unittest.main()

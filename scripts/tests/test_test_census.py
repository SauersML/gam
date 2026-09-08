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
        before = {"revision": "base", "tests": 3, "pins": Counter({"critical_12": 1})}
        after = {"tests": 4, "pins": Counter({"unrelated_13": 3})}
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            census.check_change(before, after, [])

    def test_duplicate_pin_loss_and_plain_test_count_loss_are_visible_2818(self):
        before = {"revision": "base", "tests": 5, "pins": Counter({"critical_12": 2})}
        after = {"tests": 4, "pins": Counter({"critical_12": 1})}
        self.assertEqual(census.difference(before, after), {
            "test_count_decrease": 1, "removed_pins": {"critical_12": 1}})
        with self.assertRaises(ValueError):
            census.check_change(before, after, [])

    def test_total_count_loss_fails_even_when_every_pin_survives_2818(self):
        before = {"revision": "base", "tests": 5, "pins": Counter({"critical_12": 1})}
        after = {"tests": 4, "pins": Counter({"critical_12": 1})}
        with self.assertRaisesRegex(ValueError, "without exactly one"):
            census.check_change(before, after, [])

    def test_unchanged_identities_need_no_removal_acknowledgement_2818(self):
        before = {"revision": "base", "tests": 5, "pins": Counter({"critical_12": 1})}
        after = {"tests": 6, "pins": Counter({"critical_12": 1, "new_13": 1})}
        self.assertEqual(census.check_change(before, after, []), {
            "test_count_decrease": 0, "removed_pins": {}})

    def test_acknowledgement_requires_exact_loss_and_semantic_evidence_2818(self):
        before = {"revision": "base", "tests": 2, "pins": Counter({"critical_12": 1})}
        after = {"tests": 1, "pins": Counter()}
        entry = {"base": "base", "test_count_decrease": 1,
                 "removed_pins": {"critical_12": 1}, "reason": "Replacement covers the same derivative",
                 "evidence": "replacement_12 executed: 1 passed"}
        self.assertEqual(census.check_change(before, after, [entry]), census.difference(before, after))
        for field, value in (("base", "stale"), ("test_count_decrease", 2), ("evidence", ""), ("removed_pins", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                census.check_change(before, after, [dict(entry, **{field: value})])


if __name__ == "__main__":
    unittest.main()

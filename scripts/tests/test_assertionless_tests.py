"""Unit tests for the assertion-less `#[test]` gate.

Each case pins one way the detector can be wrong. The two directions are not
symmetric and the tests say which is which: a false NEGATIVE lets one weak test
through, a false POSITIVE breaks CI for a correct test and teaches authors to
write ceremonial assertions, which would defeat the gate.
"""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assertionless_tests as gate  # noqa: E402


def names(offenders):
    return [sig for _line, sig in offenders]


class DetectsTheRealThing(unittest.TestCase):
    def test_a_print_only_test_is_flagged(self):
        src = """
#[test]
fn print_only() {
    let x = compute();
    println!("x = {x}");
}
"""
        self.assertEqual(len(gate.assertionless_tests(src)), 1)

    def test_an_asserting_test_is_not_flagged(self):
        src = """
#[test]
fn checks() {
    assert_eq!(compute(), 3);
}
"""
        self.assertEqual(gate.assertionless_tests(src), [])

    def test_a_propagating_question_counts(self):
        src = """
#[test]
fn propagates() -> Result<(), Box<dyn std::error::Error>> {
    let v = fallible()?;
    println!("{v}");
    Ok(())
}
"""
        self.assertEqual(gate.assertionless_tests(src), [])

    def test_expect_and_unwrap_do_not_count(self):
        """Inherited from the original gate, and load-bearing: `.expect` is
        overwhelmingly fixture plumbing, and counting it would clear nearly
        every offender this gate exists to catch."""
        src = """
#[test]
fn only_unwraps() {
    let v = fallible().expect("fixture");
    let w = other().unwrap();
    println!("{v} {w}");
}
"""
        self.assertEqual(len(gate.assertionless_tests(src)), 1)


class ShouldPanicIsTheAssertion(unittest.TestCase):
    def test_should_panic_after_test_attribute(self):
        src = """
#[test]
#[should_panic]
fn refuses() {
    trigger();
}
"""
        self.assertEqual(gate.assertionless_tests(src), [])

    def test_should_panic_before_test_attribute(self):
        """Attribute ORDER must not decide the verdict."""
        src = """
#[should_panic(expected = "boom")]
#[test]
fn refuses() {
    trigger();
}
"""
        self.assertEqual(gate.assertionless_tests(src), [])


class LiteralsAndCommentsAreNotCode(unittest.TestCase):
    def test_assert_inside_a_string_does_not_count(self):
        """A guard's own fixture quotes assertion syntax; so does any test that
        builds Rust source as data. Counting a quoted `assert!` would clear the
        exact tests most likely to be scanning for it."""
        src = '''
#[test]
fn builds_source_as_data() {
    let sample = "#[test] fn x() { assert!(true); }";
    println!("{sample}");
}
'''
        self.assertEqual(len(gate.assertionless_tests(src)), 1)

    def test_assert_inside_a_raw_string_with_hashes_does_not_count(self):
        src = '''
#[test]
fn raw_sample() {
    let sample = r#"assert_eq!(a, b); { unbalanced"#;
    println!("{sample}");
}
'''
        self.assertEqual(len(gate.assertionless_tests(src)), 1)

    def test_assert_inside_a_line_comment_does_not_count(self):
        src = """
#[test]
fn commented_out() {
    // assert_eq!(compute(), 3);
    println!("nothing");
}
"""
        self.assertEqual(len(gate.assertionless_tests(src)), 1)

    def test_assert_inside_a_block_comment_does_not_count(self):
        src = """
#[test]
fn commented_out_block() {
    /* assert_eq!(compute(), 3);
       still commented */
    println!("nothing");
}
"""
        self.assertEqual(len(gate.assertionless_tests(src)), 1)

    def test_a_byte_literal_brace_does_not_break_body_matching(self):
        """`b'{'` must not be counted as an opening brace, or the body span
        runs past the function and the next test's assertions clear this one."""
        src = """
#[test]
fn scans_bytes() {
    for c in text.bytes() {
        if c == b'{' { depth += 1; }
        if c == b'}' { depth -= 1; }
    }
    println!("{depth}");
}

#[test]
fn asserts_something() {
    assert!(depth == 0);
}
"""
        offenders = gate.assertionless_tests(src)
        self.assertEqual(len(offenders), 1)
        self.assertIn("scans_bytes", names(offenders)[0])


class Delegation(unittest.TestCase):
    def test_a_local_helper_that_asserts_clears_the_test(self):
        src = """
fn check_it(v: usize) {
    assert_eq!(v, 3);
}

#[test]
fn delegates_locally() {
    check_it(compute());
}
"""
        self.assertEqual(gate.assertionless_tests(src), [])

    def test_a_helper_in_another_file_clears_the_test(self):
        """The `SpeedGate` case. `sls_codegen_perf.rs` ends
        `gate.faster(...); gate.finish();` and `SpeedGate::finish` lives in
        `crates/gam-math/src/paired_timing.rs` and asserts. A file-local
        recognizer calls that test assertion-less, which is WRONG."""
        helper = """
impl SpeedGate {
    pub fn finish(mut self) {
        assert!(self.cells > 0, "a gate with no measured cell verifies nothing");
    }
}
"""
        test_src = """
#[test]
fn uses_a_gate() {
    let mut gate = SpeedGate::open("X");
    gate.finish();
}
"""
        sources = {
            "crates/other/src/gate.rs": gate.strip_source(helper),
            "crates/mine/tests/perf.rs": gate.strip_source(test_src),
        }
        impls = gate.build_impl_index(sources)
        self.assertEqual(
            gate.assertionless_tests(test_src, impls, sources, "crates/mine/tests/perf.rs"),
            [],
        )

    def test_without_the_impl_index_the_same_test_is_a_false_positive(self):
        """Pins WHY the receiver-type channel exists: this is the exact false
        positive it removes. If this case ever stops being flagged without an
        index, the file-local path has silently changed."""
        test_src = """
#[test]
fn uses_a_gate() {
    let mut gate = SpeedGate::open("X");
    gate.finish();
}
"""
        self.assertEqual(len(gate.assertionless_tests(test_src)), 1)


class PositiveControl(unittest.TestCase):
    def test_the_2110_incident_is_still_detected(self):
        """A gate that finds nothing and a gate that has stopped detecting
        anything print the same thing. This is the only thing that separates
        them: re-measure a tree that is known to be bad."""
        repo = Path(__file__).resolve().parents[2]
        try:
            blob = subprocess.run(
                ["git", "show", f"{gate.POSITIVE_CONTROL_SHA}:{gate.POSITIVE_CONTROL_PATH}"],
                cwd=repo, capture_output=True, text=True, check=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            self.fail(
                f"the positive control's blob is unreachable ({exc}); a gate whose "
                f"control cannot run has not passed"
            )
        found = gate.assertionless_tests(blob, rel=gate.POSITIVE_CONTROL_PATH)
        self.assertEqual(
            len(found), gate.POSITIVE_CONTROL_EXPECTED,
            f"issue #2110 records exactly {gate.POSITIVE_CONTROL_EXPECTED} assertion-less "
            f"#[test]s in this file at {gate.POSITIVE_CONTROL_SHA[:9]}; detector found "
            f"{len(found)}",
        )


class ResolutionIsBoundedByTheReceiverType(unittest.TestCase):
    """Resolving a call by BARE NAME across the workspace was tried and measured:
    it dropped the tree-wide count from 21 to 3, because a common method name
    always matches some definition that asserts somewhere. These pin that the
    hole stays shut."""

    def test_a_same_named_method_on_another_type_does_not_clear_the_test(self):
        helper = """
impl SomethingElse {
    pub fn finish(self) {
        assert!(self.ok, "unrelated");
    }
}
"""
        test_src = """
#[test]
fn uses_an_unrelated_type() {
    let mut thing = MyThing::open("X");
    thing.finish();
}
"""
        sources = {
            "crates/other/src/other.rs": gate.strip_source(helper),
            "crates/mine/tests/t.rs": gate.strip_source(test_src),
        }
        impls = gate.build_impl_index(sources)
        offenders = gate.assertionless_tests(test_src, impls, sources, "crates/mine/tests/t.rs")
        self.assertEqual(len(offenders), 1)

    def test_a_swallowed_error_is_still_flagged(self):
        """`zz_probe_2644_prostate_binomial_logit` in miniature: it fits, matches
        on the outcome, and prints either way. By-name resolution cleared it."""
        helper = """
impl Fitter {
    pub fn check(&self) {
        assert!(self.converged);
    }
}
"""
        test_src = """
#[test]
fn swallows_its_own_failure() {
    let outcome = fit_from_formula("y ~ s(x)", &ds, &cfg);
    match outcome {
        Ok(_) => println!("FIT OK"),
        Err(e) => println!("FIT ERR: {e}"),
    }
}
"""
        sources = {
            "crates/other/src/f.rs": gate.strip_source(helper),
            "tests/regressions/t.rs": gate.strip_source(test_src),
        }
        impls = gate.build_impl_index(sources)
        offenders = gate.assertionless_tests(test_src, impls, sources, "tests/regressions/t.rs")
        self.assertEqual(len(offenders), 1)


class ImplIndexing(unittest.TestCase):
    def test_trait_impl_keys_on_the_implementing_type(self):
        src = gate.strip_source("""
impl Display for SpeedGate {
    fn fmt(&self) {}
}
""")
        self.assertIn("SpeedGate", gate.index_impls(src))

    def test_generic_impl_keys_on_the_type_not_the_parameter(self):
        src = gate.strip_source("""
impl<T> Wrapper<T> {
    fn get(&self) {}
}
""")
        index = gate.index_impls(src)
        self.assertIn("Wrapper", index)
        self.assertNotIn("T", index)


class StripSourceCrossesLines(unittest.TestCase):
    """Every case here was found by a false positive on the real tree, not
    invented: `multinomial_separation_arming_2612.rs` was reported as
    assertion-less while containing `assert_eq!`, because quote parity had
    desynchronised twenty lines earlier."""

    def test_a_word_ending_in_r_before_a_quote_is_not_a_raw_string_prefix(self):
        """`... saddle-escape repair",` -- the final `r` of an ordinary word
        followed by a closing quote read as the raw-string prefix `r"`, opening
        a raw string that never closed and blanking the rest of the file."""
        src = '\n'.join([
            '#[test]',
            'fn has_a_multiline_message() {',
            '    let v = compute()',
            '        .expect(',
            '            "this fixture produced none at all until the escape \\',
            '             repair",',
            '        );',
            '    assert_eq!(v, 3);',
            '}',
        ])
        self.assertEqual(gate.assertionless_tests(src), [])

    def test_a_string_that_does_not_close_on_its_line_stays_open(self):
        """A `{` inside the continuation of a multi-line string must not be
        counted as an opening brace."""
        src = '\n'.join(['let s = "opens here', '  and { has a brace";', 'let t = 1;'])
        stripped = gate.strip_source(src)
        self.assertEqual(sum(line.count('{') for line in stripped), 0)

    def test_a_block_comment_spanning_lines_is_blanked_throughout(self):
        src = '\n'.join(['/* {', '   assert!(x);', '*/', 'let y = 1;'])
        stripped = gate.strip_source(src)
        self.assertFalse(any(gate.line_is_assertion_shaped(line) for line in stripped))
        self.assertEqual(sum(line.count('{') for line in stripped), 0)

    def test_a_lifetime_is_not_a_char_literal(self):
        stripped = gate.strip_source("fn f<'a>(s: &'a str) -> &'a str { s }")
        self.assertEqual(sum(line.count('{') for line in stripped), 1)

    def test_columns_are_preserved(self):
        """Reported line numbers and body spans both depend on it."""
        src = '\n'.join(['let a = "xyz"; // tail', 'let b = 2;'])
        for raw, stripped in zip(src.split('\n'), gate.strip_source(src)):
            self.assertEqual(len(raw), len(stripped))


if __name__ == "__main__":
    unittest.main()

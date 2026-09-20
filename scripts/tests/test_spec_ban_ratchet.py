import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("spec_ban_ratchet", REPO / "scripts" / "spec_ban_ratchet.py")
ratchet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ratchet)

ISSUE = "https://github.com/SauersML/gam/issues/1"


def tokens(rule, rust):
    """Tokens `rule` reports for production Rust source (test regions masked)."""
    stripped = ratchet.strip_rust(textwrap.dedent(rust)).split("\n")
    text = "\n".join("" if t else s for s, t in zip(stripped, ratchet.test_mask(stripped)))
    return [t for _, t in ratchet.RUST_RULES[rule](text)]


def write(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def commit(root, files):
    write(root, files)
    git(root, "add", "-A")
    git(root, "-c", "user.name=ratchet", "-c", "user.email=ratchet@example.invalid", "commit", "-q", "-m", "rev")


def check(root, base=None):
    out, err = io.StringIO(), io.StringIO()
    try:
        rc = ratchet.run_check(root, root / ratchet.LEDGER_REL, base, out=out, err=err)
    except ratchet.CannotMeasure as exc:
        return 2, str(exc)
    return rc, err.getvalue()


VIOLATING_LIB = """
    pub fn step(raw: f64) -> f64 {
        if raw.abs() < 1e-9 { return 0.0; }
        raw
    }
"""
LEDGER_LINE = f"magic\tcrates/demo/src/lib.rs\t< 1e-9\t{ISSUE}\n"


class Stripping(unittest.TestCase):
    def test_comments_strings_and_chars_are_blanked_with_columns_kept(self):
        src = 'let a = "x < 1e-9"; // g < 1e-9\n/* outer /* inner */ < 1e-9 */ let c = \'{\';\nlet r = r#"1e-9 "q""#;'
        out = ratchet.strip_rust(src)
        self.assertEqual([len(l) for l in out.split("\n")], [len(l) for l in src.split("\n")])
        self.assertNotIn("1e-9", out)
        self.assertNotIn("{", out)
        self.assertIn("let c", out)

    def test_lifetimes_are_not_char_literals(self):
        out = ratchet.strip_rust("fn f<'a>(x: &'a [f64]) -> f64 { x[0] < 1e-9 }")
        self.assertIn("1e-9", out)


class TestMask(unittest.TestCase):
    def test_cfg_test_module_is_masked_and_cfg_not_test_is_production(self):
        src = textwrap.dedent("""
            #[cfg(not(test))]
            fn live() { let a = 1; }
            #[cfg(test)]
            mod tests {
                fn t() {}
            }
            fn after() {}
        """).split("\n")
        mask = ratchet.test_mask(ratchet.strip_rust("\n".join(src)).split("\n"))
        masked = {line.strip() for line, t in zip(src, mask) if t}
        self.assertIn("fn t() {}", masked)
        self.assertNotIn("fn live() { let a = 1; }", masked)
        self.assertNotIn("fn after() {}", masked)

    def test_gated_mod_declaration_gates_only_itself(self):
        src = ["#[cfg(test)]", "mod tests;", "fn live() {", "}"]
        self.assertEqual(ratchet.test_mask(src), [True, True, False, False])

    def test_inner_cfg_test_masks_whole_file(self):
        self.assertEqual(ratchet.test_mask(["#![cfg(test)]", "fn f() {}"]), [True, True])

    def test_test_gated_module_file_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, {
                "crates/demo/src/lib.rs": "#[cfg(test)]\nmod probe;\nmod live;\n",
                "crates/demo/src/probe.rs": "mod deeper;\npub fn f(g: f64) -> bool { g < 1e-9 }\n",
                "crates/demo/src/probe/deeper.rs": "pub fn f(g: f64) -> bool { g < 1e-9 }\n",
                "crates/demo/src/live.rs": "pub fn f(g: f64) -> bool { g < 1e-9 }\n",
                "crates/gam-test-support/src/lib.rs": "pub fn f(g: f64) -> bool { g < 1e-9 }\n",
            })
            self.assertEqual(sorted(ratchet.production_rust(root)),
                             ["crates/demo/src/lib.rs", "crates/demo/src/live.rs"])


class Rules(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(tokens("grid", "fn f() { let seeds = [-2.0, 0.0, 2.0]; for s in &seeds { g(s); } }"),
                         ["let seeds"])
        self.assertEqual(tokens("grid", "fn f() { for s in [0.1, 1.0, 10.0] { g(s); } }"),
                         ["for-in [0.1,1.0,10.0]"])
        self.assertEqual(tokens("grid", "fn c() -> &'static [f64] { &[0.5, 1.0, 2.0] }"),
                         ["fn c [0.5,1.0,2.0]"])
        # integer index arrays, mathematical tables and un-iterated arrays are not candidate sets
        self.assertEqual(tokens("grid", "fn f() { let idx = [0, 1, 2]; for i in &idx { g(i); } }"), [])
        self.assertEqual(tokens("grid", "fn f() { let factorials = [1.0, 1.0, 2.0]; for x in &factorials {} }"), [])
        self.assertEqual(tokens("grid", "fn f() { let v = [1.0, 2.0, 3.0]; g(&v); }"), [])

    def test_box(self):
        self.assertEqual(tokens("box", "fn f(x: f64) -> f64 { x.clamp(-MAX_LOG_STEP, MAX_LOG_STEP) }"),
                         ["clamp(+-MAX_LOG_STEP)"])
        self.assertEqual(tokens("box", "fn f(x: f64) -> f64 { x.clamp(-2.0, 2.0) }"), ["clamp(+-2.0)"])
        self.assertEqual(tokens("box", "const EFS_MAX_STEP: f64 = 5.0;"), ["const EFS_MAX_STEP"])
        self.assertEqual(tokens("box", "fn f(x: f64) -> f64 { smooth_bound_jet(x, SAS_U_CLAMP) }"),
                         ["smooth_bound_jet(SAS_U_CLAMP)"])
        # domains, not boxes: correlations, latitudes; asymmetric clamps; display code
        self.assertEqual(tokens("box", "fn f(r: f64) -> f64 { r.clamp(-1.0, 1.0) }"), [])
        self.assertEqual(tokens("box", "fn f(l: f64) -> f64 { l.clamp(-std::f64::consts::FRAC_PI_2, "
                                       "std::f64::consts::FRAC_PI_2) }"), [])
        self.assertEqual(tokens("box", "fn f(x: f64) -> f64 { x.clamp(0.0, MAX) }"), [])
        self.assertEqual(tokens("box", "fn f(x: f64) -> String { format!(\"{}\", x.clamp(-9.0, 9.0)) }"), [])
        self.assertEqual(tokens("box", "fn smooth_bound_jet(value: f64, bound: f64) -> f64 { value }"), [])

    def test_jitter(self):
        self.assertEqual(tokens("jitter", "fn f() { escalate_ridge(s, |j| solve(j)); }"), ["escalate_ridge("])
        self.assertEqual(tokens("jitter", "pub fn escalate_ridge(s: S) {}"), [])
        self.assertEqual(tokens("jitter", "const MASS_MATRIX_JITTER: f64 = 1e-5;"), ["const MASS_MATRIX_JITTER"])
        self.assertEqual(tokens("jitter", "fn f() { h[[i, i]] += jitter; }"), ["diag += jitter"])
        self.assertEqual(tokens("jitter", "fn f() { h[[i, j]] += jitter; }"), [])
        self.assertEqual(tokens("jitter", "fn f() { g[[i, i]] += 1e-10 * scale; }"), ["diag += 1e-10"])
        self.assertEqual(tokens("jitter", "fn f() { g[[i, i]] += (2.5e-8 * s); }"), ["diag += 2.5e-8"])
        self.assertEqual(tokens("jitter", "fn f() { g[[i, j]] += 1e-10; }"), [])
        self.assertEqual(tokens("jitter", "fn f() { g[[i, i]] += 2.0 * w; }"), [])

    def test_unconverged(self):
        self.assertEqual(tokens("unconverged", "fn f() -> R { Ok(Fit { beta: b, converged: false }) }"),
                         ["Ok(Fit{converged:false})"])
        self.assertEqual(tokens("unconverged", "fn f() -> R { Ok(Fit { beta: b, converged: true }) }"), [])
        self.assertEqual(tokens("unconverged", "fn new() -> R<Self> { Ok(Self { converged: false }) }"), [])
        self.assertEqual(tokens("unconverged", "fn f() -> R { Err(Fit { converged: false }) }"), [])

    def test_magic(self):
        self.assertEqual(tokens("magic", "fn f(g: f64) -> bool { g.abs() < 1e-8 }"), ["< 1e-8"])
        self.assertEqual(tokens("magic", "fn f(g: f64) -> bool { 1.0e-10 >= g }"), ["1.0e-10 >="])
        self.assertEqual(tokens("magic", "fn f(g: f64) -> f64 { g.max(1e-300) }"), [".max(1e-300)"])
        self.assertEqual(tokens("magic", "fn f(g: f64) -> bool { g == 1e-8 }"), [])
        self.assertEqual(tokens("magic", "fn f(g: u8) -> f64 { match g { 0 => 1e-8, _ => 0.0 } }"), [])
        self.assertEqual(tokens("magic", "fn f(g: f64) -> f64 { g * 1e-8 }"), [])
        self.assertEqual(tokens("magic", "fn f(g: f64) -> bool { g < f64::EPSILON }"), [])
        self.assertEqual(tokens("magic", "#[cfg(test)]\nmod t {\n    fn f(g: f64) -> bool { g < 1e-8 }\n}\n"), [])

    def test_fd(self):
        self.assertEqual(tokens("fd", "fn d(x: f64, h: f64) -> f64 { (loss(x + h) - loss(x - h)) / (2.0 * h) }"),
                         ["central loss(+-h)"])
        self.assertEqual(tokens("fd", "fn d(x: f64, h: f64) -> f64 { (loss(x + h) - loss(x)) / h }"),
                         ["forward loss(+h)"])
        # a difference of two evaluations that is not divided by the step is not a derivative
        self.assertEqual(tokens("fd", "fn d(x: f64) -> f64 { ln_gamma(x + a) - ln_gamma(x) }"), [])

    def test_gcv(self):
        self.assertEqual(tokens("gcv", "fn f() -> f64 { gcv_score(1.0) + GcvCriterion::eval() }"),
                         ["gcv_score", "GcvCriterion"])
        self.assertEqual(tokens("gcv", "fn f() -> f64 { mgcv_reference(1.0) + UBRE_WEIGHT }"), ["UBRE_WEIGHT"])

    def test_roundoff(self):
        self.assertEqual(tokens("roundoff", "fn f() -> f64 { 0.5 * f64::EPSILON }"), ["0.5 * f64::EPSILON"])
        self.assertEqual(tokens("roundoff", "fn f() -> f64 { f64::EPSILON / 2.0 }"), ["f64::EPSILON / 2.0"])
        self.assertEqual(tokens("roundoff", "fn f(x: f64) -> f64 { x * f64::EPSILON * 0.5 }"),
                         ["f64::EPSILON * 0.5"])
        self.assertEqual(tokens("roundoff", "fn g(n: f64) -> f64 { n * f64::EPSILON / (1.0 - n * f64::EPSILON) }"),
                         ["EPSILON / (1.0 -"])
        self.assertEqual(tokens("roundoff", "fn g(n: f64) -> f64 { (n * f64::EPSILON) / (1.0 - n) }"),
                         ["EPSILON) / (1.0 -"])
        # a half-log of epsilon, an unrelated multiple, and test code are not copies of `u`
        self.assertEqual(tokens("roundoff", "fn f() -> f64 { -0.5 * f64::EPSILON.ln() }"), [])
        self.assertEqual(tokens("roundoff", "fn f() -> f64 { 2.0 * f64::EPSILON + f64::EPSILON / 20.0 }"), [])
        self.assertEqual(tokens("roundoff", "#[cfg(test)]\nmod t {\n    const U: f64 = 0.5 * f64::EPSILON;\n}\n"), [])

    def test_roundoff_owner_may_define_the_unit_roundoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, {
                "crates/demo/src/lib.rs": VIOLATING_LIB,
                ratchet.ROUNDOFF_OWNER: "pub const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;\n",
                ratchet.LEDGER_REL: LEDGER_LINE,
            })
            self.assertEqual(check(root)[0], 0)
            write(root, {"crates/demo/src/band.rs": "pub fn u() -> f64 { f64::EPSILON / 2.0 }\n"})
            rc, err = check(root)
            self.assertEqual(rc, 1)
            self.assertIn("[roundoff] f64::EPSILON / 2.0 -- new SPEC violation", err)

    def test_python_math(self):
        src = textwrap.dedent('''
            """Uses np.linalg.solve in a docstring."""
            import torch
            from scipy.optimize import minimize
            # np.linalg.inv(a)
            def f(a):
                s = "torch.linalg.qr"
                return torch.linalg.svd(a)
        ''')
        found = [t for _, t in ratchet.rule_python_math(ratchet.python_code_only(src))]
        self.assertEqual(found, ["from scipy.optimize import", "torch.linalg.svd"])

    def test_positive_control_passes(self):
        self.assertEqual(ratchet.positive_control(out=io.StringIO(), err=io.StringIO()), 0)


class Ledger(unittest.TestCase):
    def test_malformed_ledger_cannot_measure(self):
        for bad in ("magic\tp\tt\n", f"nope\tp\tt\t{ISSUE}\n", "magic\tp\tt\thttps://example.com/1\n"):
            with self.assertRaises(ratchet.CannotMeasure):
                ratchet.parse_ledger(bad, "ledger")

    def test_compare_reports_new_and_stale_as_multisets(self):
        hits = [("magic", "a.rs", "< 1e-9", 3, ""), ("magic", "a.rs", "< 1e-9", 9, "")]
        ledger = ratchet.parse_ledger(f"magic\ta.rs\t< 1e-9\t{ISSUE}\nbox\tb.rs\tconst X_STEP_CAP\t{ISSUE}\n", "l")
        new, stale = ratchet.compare(hits, ledger)
        self.assertEqual([h[3] for h in new], [9])
        self.assertEqual(dict(stale), {("box", "b.rs", "const X_STEP_CAP"): 1})

    def test_tree_must_equal_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, {"crates/demo/src/lib.rs": VIOLATING_LIB})
            self.assertEqual(check(root)[0], 2)  # no ledger
            write(root, {ratchet.LEDGER_REL: "# only a comment\n"})
            self.assertEqual(check(root)[0], 2)  # empty ledger certifies nothing
            write(root, {ratchet.LEDGER_REL: LEDGER_LINE})
            self.assertEqual(check(root)[0], 0)
            write(root, {"crates/demo/src/lib.rs": VIOLATING_LIB + "pub fn g(x: f64) -> f64 { x.max(1e-12) }\n"})
            rc, err = check(root)
            self.assertEqual(rc, 1)
            self.assertIn("[magic] .max(1e-12) -- new SPEC violation", err)
            write(root, {"crates/demo/src/lib.rs": "pub fn g(x: f64) -> f64 { x }\n"})
            rc, err = check(root)
            self.assertEqual(rc, 2)  # zero hits anywhere: the scanner, not the tree, is suspect
            write(root, {"crates/demo/src/other.rs": "pub fn g(x: f64) -> bool { x < 1e-3 }\n"})
            rc, err = check(root)
            self.assertEqual(rc, 1)
            self.assertIn("stale entry", err)
            self.assertIn("new SPEC violation", err)

    def test_prune_only_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, {
                "crates/demo/src/lib.rs": VIOLATING_LIB,
                ratchet.LEDGER_REL: "# header\n" + LEDGER_LINE + f"box\tcrates/demo/src/lib.rs\tclamp(+-2.0)\t{ISSUE}\n",
            })
            ratchet.prune(root, root / ratchet.LEDGER_REL)
            self.assertEqual((root / ratchet.LEDGER_REL).read_text(), "# header\n" + LEDGER_LINE)
            self.assertEqual(check(root)[0], 0)

    def test_ledger_only_shrinks_against_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            git(root, "init", "-q")
            commit(root, {"README": "x\n"})
            write(root, {"crates/demo/src/lib.rs": VIOLATING_LIB, ratchet.LEDGER_REL: LEDGER_LINE})
            rc, _ = check(root, "HEAD")
            self.assertEqual(rc, 0)  # the base predates the ratchet: introduction passes
            commit(root, {ratchet.SCRIPT_REL: "# ratchet\n", ratchet.LEDGER_REL: LEDGER_LINE,
                          "crates/demo/src/lib.rs": VIOLATING_LIB})
            self.assertEqual(check(root, "HEAD")[0], 0)

            # adding a violation together with its ledger line is refused
            grown = VIOLATING_LIB + "pub fn g(x: f64) -> f64 { x.max(1e-12) }\n"
            write(root, {"crates/demo/src/lib.rs": grown, ratchet.LEDGER_REL:
                         LEDGER_LINE + f"magic\tcrates/demo/src/lib.rs\t.max(1e-12)\t{ISSUE}\n"})
            rc, err = check(root, "HEAD")
            self.assertEqual(rc, 1)
            self.assertIn("the ledger only accepts deletions", err)

            # moving a violation to another file is a new line, so it is refused too
            write(root, {"crates/demo/src/lib.rs": "mod moved;\npub fn f(g: f64) -> f64 { g.max(1e-12) }\n",
                         "crates/demo/src/moved.rs": VIOLATING_LIB,
                         ratchet.LEDGER_REL: f"magic\tcrates/demo/src/moved.rs\t< 1e-9\t{ISSUE}\n"
                                             f"magic\tcrates/demo/src/lib.rs\t.max(1e-12)\t{ISSUE}\n"})
            self.assertEqual(check(root, "HEAD")[0], 1)

            # deleting the fixed line is accepted
            commit(root, {"crates/demo/src/lib.rs": grown, ratchet.LEDGER_REL:
                          LEDGER_LINE + f"magic\tcrates/demo/src/lib.rs\t.max(1e-12)\t{ISSUE}\n"})
            (root / "crates/demo/src/moved.rs").unlink()
            write(root, {"crates/demo/src/lib.rs": "pub fn g(x: f64) -> f64 { x.max(1e-12) }\n",
                         ratchet.LEDGER_REL: f"magic\tcrates/demo/src/lib.rs\t.max(1e-12)\t{ISSUE}\n"})
            self.assertEqual(check(root, "HEAD")[0], 0)

            # a base with the script but no ledger cannot be measured, and neither can a bogus revision
            git(root, "rm", "-q", "-f", ratchet.LEDGER_REL)
            git(root, "-c", "user.name=r", "-c", "user.email=r@example.invalid", "commit", "-q", "-m", "drop")
            write(root, {ratchet.LEDGER_REL: f"magic\tcrates/demo/src/lib.rs\t.max(1e-12)\t{ISSUE}\n"})
            self.assertEqual(check(root, "HEAD")[0], 2)
            self.assertEqual(check(root, "no-such-rev")[0], 2)


class RealTree(unittest.TestCase):
    def test_repository_agrees_with_its_ledger(self):
        rc, err = check(REPO)
        self.assertEqual(rc, 0, err)

    def test_every_ledger_line_is_a_distinct_well_formed_entry(self):
        text = (REPO / ratchet.LEDGER_REL).read_text(encoding="utf-8")
        ledger = ratchet.parse_ledger(text, ratchet.LEDGER_REL)
        self.assertGreater(sum(ledger.values()), 0)
        body = [l for l in text.split("\n") if l and not l.startswith("#")]
        self.assertEqual(body, sorted(body), "keep the ledger sorted so deletions diff cleanly")


if __name__ == "__main__":
    unittest.main()

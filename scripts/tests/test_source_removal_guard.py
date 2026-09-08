import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("source_removal_guard", Path(__file__).parents[1] / "source_removal_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


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


if __name__ == "__main__":
    unittest.main()

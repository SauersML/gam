base = open('.land993tmp/harvest_edit1.rs').read()
testblock = open('.land993tmp/test_block.txt').read()

# Insert after the byte-deterministic test's closing brace. Anchor on its
# final assert + closing brace, which appears once.
anchor = """        assert_eq!(a.term.k_atoms(), b.term.k_atoms());
    }
"""
assert base.count(anchor) == 1, f"anchor count={base.count(anchor)}"
# testblock already starts with a leading 4-space indent and a blank-ish line;
# ensure exactly one blank line separates.
replacement = anchor + "\n" + testblock
base = base.replace(anchor, replacement)
open('.land993tmp/harvest_final.rs','w').write(base)
print("final line count:", base.count(chr(10))+1)

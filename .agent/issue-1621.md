# Issue #1621 — debiased_functional point/contrast column-order bug

## Status on arrival
- Issue #1621 ALREADY FIXED + CLOSED on main (fe1425a / commit `debiased_query_design_full_schema`).
- Sibling #1622 (missing weighted Gram) also fixed + closed.
- Both regression tests present:
  - tests/bug_hunt_debiased_point_contrast_column_order_test.py
  - tests/bug_hunt_debiased_point_contrast_full_schema_test.py

## Plan
1. Verify the fix actually builds + passes (be my own CI).
2. Hunt for adjacent bugs in model_debiased_functional_json_impl neighborhood:
   - placeholder/by-column handling
   - column-order invariance edge cases (categorical predictors, by= factors)
   - average_value / average_derivative arms
3. Harden + add tests where genuinely valuable.

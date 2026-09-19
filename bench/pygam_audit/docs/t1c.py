import numpy as np, pandas as pd, gamfit, warnings, re
warnings.simplefilter("ignore")
df = pd.read_csv("data/Wage.csv")[["year", "age", "education", "wage"]]
D = {c: df[c].to_numpy() for c in df}
m = gamfit.fit(D, "wage ~ s(year) + s(age) + education")
s = m.summary()
print(s.smooth_terms_frame())
print(s.coefficients_frame().head(8))
print(repr(s))
h = s._repr_html_(); print("html len", len(h), "has p-value col:", "p" in h and "value" in h.lower())
print(re.sub("<[^>]+>", " ", h)[:1500])
print(m.term_blocks)
try:
    print(m.smooth_significance())
except Exception as e: print("smooth_significance:", e)

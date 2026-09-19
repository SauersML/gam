import numpy as np, pandas as pd, gamfit, warnings
warnings.simplefilter("ignore")
df = pd.read_csv("data/Wage.csv")[["year", "age", "education", "wage"]]
D = {c: df[c].to_numpy() for c in df}
m = gamfit.fit(D, "wage ~ s(year) + s(age) + education")
s = m.summary()
print(type(s)); print([k for k in s.to_dict().keys()])
for k in ["term_tests", "terms", "edf", "smooth_terms", "term_edf", "per_term"]:
    if k in s.to_dict(): print(k, s.to_dict()[k])
print(s.coefficients[:3])
print([c.get("name") or c.get("term") for c in s.coefficients])
print([a for a in dir(m) if not a.startswith("_")])
print([a for a in dir(s) if not a.startswith("_")])

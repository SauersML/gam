import numpy as np, pandas as pd, gamfit, warnings
warnings.simplefilter("ignore")
df = pd.read_csv("data/Wage.csv")[["year", "age", "education", "wage"]]
D = {c: df[c].to_numpy() for c in df}
for f in ["wage ~ s(age) + education", "wage ~ s(age) + factor(education)", "wage ~ s(age) + group(education)"]:
    m = gamfit.fit(D, f)
    print(f, [ (b.name,b.kind) for b in m.term_blocks], m.summary().lambdas)
    try:
        new = {"age": np.array([40,40.]), "education": np.array(["9. Unseen", "1. < HS Grad"])}
        print("  unseen level predict:", m.predict(new))
    except Exception as e:
        print("  unseen level predict raises:", type(e).__name__, str(e)[:100])

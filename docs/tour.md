# A tour on real data

Six short analyses on public datasets, each a complete program: copy a
block into a fresh Python session and it runs. The data come from the
[Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) mirror, so
every block needs network access the first time it runs.

No block sets a smoothing parameter, a number of splines or a search
grid. REML/LAML chooses every smoothing parameter in one optimization,
and the printed effective degrees of freedom (edf) show what it chose.

gamfit announces the basis size it picked for each smooth with a
`GamInferenceWarning`; see [Choosing `k`](formulas.md#choosing-k) for
what that size means.

## Wages: smooths, a factor and model comparison

The ISLR `Wage` data: 3,000 male workers in the US Mid-Atlantic region.
Wage rises with age and then falls, rises slowly over the survey years,
and differs by education level.

```python
import pandas as pd
import gamfit

wage = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Wage.csv")

candidates = {
    "age only": "wage ~ s(age)",
    "+ education": "wage ~ s(age) + education",
    "+ year": "wage ~ s(age) + s(year, k=5) + education",
}
fits = [gamfit.fit(wage, formula) for formula in candidates.values()]

comparison = gamfit.compare_models(fits, names=list(candidates))
print(comparison["evidence_summary"])
for name, score, delta, evidence_ratio, edf in comparison["ranking"]:
    print(f"{name:12s}  delta={delta:8.2f}  evidence ratio={evidence_ratio:.3g}  edf={edf:.2f}")

print(fits[-1].summary().smooth_terms_frame())
```

`compare_models` ranks the fits by conditional AIC and reports each
one's Akaike evidence ratio against the winner; the full model wins by a
ratio of several hundred over the model without `year`. The term table
shows `s(year, k=5)` with an edf near 1: `year` has only seven distinct
values, and REML shrank its smooth to an almost straight line instead of
chasing them. `education` is a string column, so it enters as a factor
(see [categorical terms](formulas.md#factor-terms)).

## Heteroscedastic noise (mcycle)

The MASS `mcycle` data: head acceleration of a crash-test dummy in the
milliseconds after a simulated motorcycle impact. The noise is small
before the impact and large during it, so one noise level fits no part of
the curve well. `noise_formula=` makes the noise scale a smooth of time
as well, fitted jointly with the mean.

```python
import pandas as pd
import gamfit

mcycle = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv")

constant_noise = gamfit.fit(mcycle, "accel ~ s(times)")
smooth_noise = gamfit.fit(mcycle, "accel ~ s(times)", noise_formula="s(times)")

for label, model in [("constant noise", constant_noise), ("s(times) noise", smooth_noise)]:
    bands = model.predict(mcycle, interval=0.95, observation_interval=True)
    inside = mcycle["accel"].between(bands["observation_lower"], bands["observation_upper"])
    early = mcycle["times"] < 12
    width = bands["observation_upper"] - bands["observation_lower"]
    print(f"{label:15s} coverage={inside.mean():.1%}  "
          f"interval width before 12 ms={width[early].mean():6.1f} g, after={width[~early].mean():6.1f} g")
```

Both 95% observation intervals cover 95–97% of the points overall, but
the constant-noise interval is just as wide before the impact as
during it. The location-scale fit's interval is narrow where the data are
quiet and wide where they are not, which is the figure at the top of the
[README](https://github.com/SauersML/gam#readme). The prediction frame
also carries the fitted `noise_scale` for each row.

## Counts: Poisson, overdispersion and the negative binomial

The carData `Ornstein` data: the number of interlocking directorships of
248 Canadian firms, with each firm's assets, sector and nation of
control. The response is a non-negative integer, so the default family is
Poisson with a log link.

```python
import numpy as np
import pandas as pd
import gamfit

firms = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/carData/Ornstein.csv")
firms["log_assets"] = np.log(firms["assets"])
formula = "interlocks ~ s(log_assets) + sector + nation"

poisson = gamfit.fit(firms, formula)
negbin = gamfit.fit(firms, formula, family="negative-binomial")

for model in (poisson, negbin):
    check = model.basis_check(firms)[0]
    print(f"{model.summary().family_name:22s} edf of s(log_assets)={check['edf']:.2f}  "
          f"basis check p={check['p_value']:.2g}")
```

The Poisson fit spends about seven degrees of freedom on
`s(log_assets)`, and `basis_check` reports structure left in its
residuals (a tiny p-value). The counts vary far more than a Poisson
allows, and the smooth is soaking up that extra variance as wiggles. The
negative binomial fit estimates the overdispersion. Its smooth relaxes to
nearly a straight line in log assets (edf about 1), and its basis check
finds nothing left over. `compare_models` refuses to rank these two fits
against each other, because Poisson and negative binomial likelihoods are
on different base measures; the residual check is the evidence here.

## Binary outcomes with string labels (Mroz)

The carData `Mroz` data: labour-force participation of 753 married women
in 1975, recorded as the strings `"no"` and `"yes"`. A string response
needs `family="binomial"`, and the modelled event is the label that sorts
last in plain string order, here `"yes"`. That is the same class
[`GAMClassifier`](sklearn.md#gamclassifier) reports as `classes_[1]`.

```python
import pandas as pd
import gamfit

mroz = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/carData/Mroz.csv")

model = gamfit.fit(mroz, "lfp ~ s(age) + s(inc) + k5 + wc", family="binomial")

p_yes = model.predict(mroz)
print(f"mean predicted P(lfp == 'yes') = {p_yes.mean():.3f}, "
      f"observed share = {(mroz['lfp'] == 'yes').mean():.3f}")
print(model.summary().smooth_terms_frame())
```

The mean predicted probability matches the observed share of `"yes"`.
REML keeps `s(age)` at an edf near 1, a straight line on the logit scale,
and gives `s(inc)` a gentle bend. If the event you care about sorts
first, recode the column (for example to `0`/`1`) before fitting.

## Shape constraints: a monotone curve (cars)

The `cars` data: stopping distance against speed for 50 cars from the
1920s. Stopping distance cannot fall as speed rises, and a
`shape=monotone_increasing` smooth builds that knowledge into the fit.

```python
import numpy as np
import pandas as pd
import gamfit

cars = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/cars.csv")

model = gamfit.fit(cars, "dist ~ s(speed, shape=monotone_increasing)")

grid = pd.DataFrame({"speed": np.linspace(cars["speed"].min(), cars["speed"].max(), 200)})
curve = model.predict(grid)
print("never decreases:", bool(np.all(np.diff(curve) >= 0)))
print(f"stopping distance at 10 mph = {np.interp(10, grid['speed'], curve):.1f} ft, "
      f"at 20 mph = {np.interp(20, grid['speed'], curve):.1f} ft")
```

`shape=` also accepts `monotone_decreasing`, `convex` and `concave`; see
[shape constraints](formulas.md#shape-constraints).

## Surfaces: a tensor product (topo)

The MASS `topo` data: 52 elevation readings over a 6.5 × 6.5 plot. A
tensor-product smooth `te(x, y)` builds a surface from one marginal basis
per coordinate, each with its own smoothing parameter, so it suits
covariates on different scales as well as spatial data like this.

```python
import numpy as np
import pandas as pd
import gamfit

topo = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/topo.csv")

model = gamfit.fit(topo, "z ~ te(x, y)")
print(model.summary().smooth_terms_frame())

grid = pd.DataFrame({"x": np.repeat(np.linspace(0, 6.5, 5), 5),
                     "y": np.tile(np.linspace(0, 6.5, 5), 5)})
surface = model.predict(grid).reshape(5, 5)
print(np.round(surface))
```

The rows of the printed grid run along `x` and the columns along `y`. The
[multivariate smooths](formulas.md#multivariate-smooths) section covers
isotropic alternatives (`s(x, y)`, `matern(x, y)`, `duchon(x, y)`) for
when both coordinates share one scale.

## Where next

- [Migrating from pyGAM](migrating-from-pygam.md) maps pyGAM calls to gamfit.
- [Predictions](predictions.md) lists every column `predict` can return.
- [Diagnostics](diagnostics.md) covers `summary()`, `basis_check()` and plots.
- [Benchmarks](benchmarks.md) compares accuracy and speed with pyGAM, losses included.

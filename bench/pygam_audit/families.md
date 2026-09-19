# Audit axis: response families and link functions

gamfit 0.1.267 wheel (repo HEAD d10950d, pyproject 0.1.268) vs pyGAM 0.12.0.
Scripts and raw outputs: `scratchpad/audit/families/`
(`probe_api.py`, `compare_fits.py`, `probe_sas.py`, `probe_pygam_binom.py`,
`probe_pygam_cells.py`, `probe_pygam_exposure_gs.py`, `probe_expectile.py`,
`compare_out_part1.txt`, `compare_out2.txt`, `expectile_out.txt`).
Python was run from the scratchpad (running from `/home/user/gam` imports the
source tree, which has no `_rust`). gamfit was fed dict-of-numpy input
(pandas input fails without pyarrow: `ImportError: Import pyarrow failed`,
not this axis). No `gam` CLI binary is installed, so CLI parity is judged
from source (`crates/gam-cli`).

## 1. Coverage matrix (pyGAM feature -> gamfit)

| pyGAM feature | gamfit Python | formula DSL | CLI | Works in practice? |
|---|---|---|---|---|
| LinearGAM (normal/identity) | `family="gaussian"` | - | `--family gaussian` | yes, better than pyGAM (table 2) |
| LogisticGAM (binomial/logit) | `family="binomial"` | `link(type=logit)` | `--family binomial-logit` | yes, better |
| binomial with trials/levels | proportions `y` + `weights=` | - | `--weights` | yes, better (pyGAM per-row levels is broken, see F8) |
| PoissonGAM | `family="poisson"` | - | `--family poisson-log` | yes |
| PoissonGAM exposure | `offset="log_e_column"` | `offset(...)` | `--offset` | yes, better (pyGAM gridsearch+exposure is broken, F7) |
| GammaGAM (default link log) | `family="gamma"` (log) | - | `--family gamma-log` | yes (table 2) |
| GAM(gamma, link="inverse") (canonical) | NO: `unsupported link type 'inverse'` | NO | NO | gap F2 |
| InvGaussGAM | NO: `unknown family 'inverse-gaussian'` | NO | NO | gap F1 |
| ExpectileGAM | `family="expectile"`, `expectile_tau=` or `expectile(0.9)` | - | `--family expectile --expectile-tau` | yes (LAWS + REML), see F5 |
| ExpectileGAM.fit_quantile | NO (by design; CTN quantiles exist) | - | - | pyGAM slop, F6 |
| GAM(normal, link=log / inverse / inv_squared) | NO | NO | NO | gap F2 |
| GAM(poisson, identity) | NO (rejected by legality matrix) | - | - | gap F2 (pyGAM accepts it unconstrained, F9) |
| GAM(gamma, identity) | NO | - | - | gap F2 |
| GAM(binomial, identity) | NO | - | - | pyGAM accepts silently, F9; gamfit rejection is correct |
| sample weights | `weights=` | - | `--weights` | yes (pyGAM casts to float32, F7) |
| not in pyGAM: probit, cloglog, loglog, cauchit, beta-logistic, latent-cloglog, link mixtures, SAS | yes | yes | yes | yes except SAS (bug F4) |
| not in pyGAM: negative binomial, beta, tweedie, multinomial, Royston-Parmar survival, transformation-normal | yes | - | yes | already-better F10 |

Evidence of the pinned-link legality matrix:
`crates/gam-spec/src/lib.rs:1457-1491` (`is_legal_cell`: Gaussian/Royston-Parmar
only Identity; Poisson/Gamma/Tweedie/NB only Log; Beta only Logit; Binomial all
probability links but not Identity/Log), mirrored in
`crates/gam-models/src/fit_orchestration/materialize/family.rs:56-76`
(`link_legal_for_family`). `LinkFunction` at `crates/gam-spec/src/lib.rs:116-126`
has no Inverse/InverseSquared/Sqrt variant; `ResponseFamily` at
`crates/gam-spec/src/lib.rs:605-631` has no InverseGaussian.

`probe_api.py` output (abridged):
```
FAIL gauss {'family': 'gaussian', 'link': 'log'} -> link 'log' is not supported for family 'gaussian'
FAIL gauss {'family': 'gaussian', 'link': 'inverse'} -> unsupported link type 'inverse'; use one of identity|log|logit|probit|cloglog|loglog|cauchit|...
FAIL pos   {'family': 'gamma', 'link': 'inverse'} -> unsupported link type 'inverse'
FAIL pos   {'family': 'gamma', 'link': 'identity'} -> not supported
FAIL pos   {'family': 'gamma(inverse)'} -> ... use one of identity|log|logit|probit|cloglog|sas|beta-logistic   (vocabulary differs from the one above)
FAIL pos   {'family': 'inverse-gaussian' | 'inverse_gaussian' | 'inv_gauss' | 'invgauss'} -> unknown family
FAIL count {'family': 'poisson', 'link': 'identity'} -> not supported
FAIL count {'family': 'poisson', 'link': 'sqrt'} -> unsupported link type
OK   bin   binomial logit / probit / cloglog / loglog / cauchit, binomial(cauchit)
FAIL bin   {'family': 'binomial', 'link': 'log'} -> relative-risk link not supported
FAIL bin   {'family': 'binomial', 'link': 'sas'} -> (see F4)
OK   gauss expectile, expectile_tau=0.9  and  family='expectile(0.9)'   (family_name reports 'Gaussian Identity')
FAIL gauss {'family': 'quantile'} -> unknown family
```

## 2. Head-to-head fit quality

Truth `eta = sin(2 pi x1) + 0.8*4(x2-0.5)^2 - 0.27`, formula `y ~ s(x1) + s(x2)`,
pyGAM `s(0) + s(1)`. RMSE of the fitted mean on the response scale against the
known truth on a 60x9 held-out grid; `cover95` = fraction of grid points whose
95% interval for the mean contains the truth. pyGAM default = fixed lam=0.6;
pyGAM gridsearch = its recommended 11-point lam grid on GCV/UBRE. The machine
was heavily oversubscribed (load average 40+ on 4 CPUs), so absolute times
are inflated; only ratios within a row block are meaningful.

RESULTS_TABLE

Reading: on every family gamfit supports, gamfit's REML/LAML fit is at least
as accurate as pyGAM's best (gridsearch) and its intervals are calibrated or
slightly conservative, whereas pyGAM gridsearch intervals under-cover
(0.88-0.94). pyGAM default (lam=0.6) is uniformly less accurate.

## 3. Findings

FINDINGS_BODY

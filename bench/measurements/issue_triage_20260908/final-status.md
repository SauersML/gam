Verified issue closures published on main: #2830, #2831, #2833, #2835, #2748.

MSI verification includes 146 block/constrained REML cases, 30 Firth seeds
and three controls, seven final Python API/persistence tests, 34 survival
derivative tests, and three spatial basis derivative tests. These populations
overlap in their implementation coverage and are not a workspace-wide census.

The spatial fixes are f20857495 (logarithmic kernel derivatives) and fba2bc637
(literal zero anisotropy). Their basis tests pass. The model-level gradient
check failed before the second fix; its final rerun is unverified. MSI became
unreachable during the final canonical models-unit compilation. The preceding
survival and full-size spatial integration targets compiled successfully in
9.40 and 5.12 seconds, respectively. Their final full-fit acceptance was not run.

Eight eligible issues remain open: #2834, #2817, #2767, #2735, #2714, #2695,
#1561, and #1082. This work does not establish completion of the original
all-issues request. GPU/SAE issues, including mixed-scope issues, were excluded;
#2469's repository-wide inventory explicitly includes both excluded components.

Unpublished follow-up work remains in the shared working tree:

- #2817: criterion-resolution decrement hook and its public regression; the
  regression passes on the recorded working-tree graph, but publication needs
  the complete criterion-resolution API dependency set and original-size fit.
- #2834: Cholesky roundoff admission correction and regression; production
  compiled, but final public acceptance is unverified.
- #2735: model-level component regression; final execution was interrupted.

The installed verified Python extension remains 7112c1fffe857fb5eaf709c93bbc6de6b899925329f287b53ef66ba506c7509c.
Optimized native libraries were preserved, but final wheel linking did not
finish inside the build bound. The last models-unit process status could not
be confirmed after the MSI connection loss. Further investigations and broad
rebuilds were stopped when the user requested that the work be wrapped up.

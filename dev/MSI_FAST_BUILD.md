# Fast Rust build/iteration on MSI

Owner: build-speed agent. Measured 2026-06-17 on a 32-core `msismall` node,
full `cargo test --no-run` (18 integration-test bins + ~80 deps, profile.test
`opt-level=2` for all deps).

## Numbers

| build | wallclock |
|-------|-----------|
| cold, GNU ld, no cache | **357 s** |
| cold, **mold** linker | **303 s**  (~15% off — pure win, all in the link) |
| fresh target dir, warm sccache | ~314 s — **0.00% Rust cache hits** |

sccache was evaluated twice (incl. `CARGO_INCREMENTAL=0`): C/C++ objects cache
100% cross-target, but **Rust rlibs get 0% cross-target-dir hits** (sccache
keys on the per-target `--extern` / `-L dependency=<abs>/deps` paths). So
sccache does NOT let two different target dirs share dep compiles here. It
stays **opt-in** (`gam_sccache_on`), useful only for a clean rebuild in the
*same* target dir.

## The recipe (all wired in `gam_env.sh`)

1. **Own target dir** — cargo locks a target dir exclusively, so many agents on
   one dir serialize. Set `GAM_TARGET_TAG=<issue>` before sourcing; you get
   `/scratch.global/sauer354/gam-target-<issue>`. Pay the ~300 s cold dep build
   ONCE per tag, then every rebuild in that dir is warm/incremental.
2. **mold linker** — auto-wired via `gcc -B<.local/libexec/mold>` (gcc 8.5 is
   too old for `-fuse-ld=PATH`). ~15% off every build.
3. **Build once, run the binary** — don't `cargo test <name>` in a loop (it
   re-walks fingerprints each call). Use the helper:

   ```
   source /projects/standard/hsiehph/sauer354/gam_env.sh
   gam_test <test_file_stem> [name_substring]   # builds bin once, runs it directly
   ```

4. **Keep-warm** — `msi sub dev/keep_warm.sbatch` (or the copy at
   `/projects/standard/hsiehph/sauer354/keep_warm.sbatch`) ff-pulls origin/main
   and rebuilds into a dedicated warm target so the next job is incremental.
   Forward-only; never `git reset --hard` (busts fingerprints, re-colds the
   fleet).

## One-liner to run a single named test fast

```
source /projects/standard/hsiehph/sauer354/gam_env.sh
GAM_TARGET_TAG=myissue   # or export before sourcing
gam_test objective_gradient_consistency_universal binomial_logit
```

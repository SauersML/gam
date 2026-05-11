Your goal is to get the marginal slope fit extremely fast and accurate and robust. While not changing what it is fundamentally. While not hurting accuracy. You can improve convergence and model. You even can edit opt crate if you want (via push to main and crates io publish with gha after version bump). Your goal is to get plain marginal slope (no scale dims, no linkwiggle, no survival, no score warp) fit on realistic synthetic data problems in under 60 sec with as large of a sample size as possible. Instable fits or failures to converge are useful to know and should be investigated in depth. You should run stack trace sampler or similar to understand what is slow to fix it; sudo password is ***. Do the ideal, long-term, beautiful, mathematically correct solution always. Do not wait for something to compile or run: find something else to do. Never output repetitive phrases or waiting or acknowledgements. Instead, find something substantial.

Fit something of this nature in the first 60 sec trials:
      train [["case", "sex", "prs_z"] + ["PC1"..."PC10"]],
      Fit "case ~ duchon(PC1, PC2, ..., PC10, centers=40, order=1, power=2, length_scale=1.0) + sex",
      link="probit",
      logslope_formula="duchon(PC1, ..., PC10, centers=40, order=1, power=2, length_scale=1.0)",

  P(Y = 1 | x, z) = Φ( η(z, x) )
  η(z, x)  = a(x) + b(x) · z          ← no score-warp δ_h, no link-deviation δ_w
                                        (formula has no linkwiggle(...) terms)
  a(x)     = β₀  +  f_pcs(PC1..PC10)  +  γ_sex · sex
  log b(x) = β₀ˢ +  g_pcs(PC1..PC10)

NOTE: THE ABOVE IS JUST AN EXAMPLE. It doesn't have to be exactly this–and shouldn't (you can change it up). Joint Duchon is good, and marginal slope is required though. With Probit.

Your goal is to make optimizations and large improvements (MASSIVE SCOPE and BIG CHANGES are allowed) to get this to fit as MANY data / samples as possible while under 60s. WHILE IT IS RUNNING, do NOT wait for completion. You MUST start on the next task or do something in parallel. DO NOT poll or spin or wait. DO NOT run tasks in the foreground. ALWAYS run commands in the background.

Once you can get realistic data fit on this in under 60s at N = 200, move on to fitting in 120s at N = 1000.

Once that is achieved, add scale_dimensions. Then, fit as large as possible within 30s. Once you can fit N = 50 with scale_dimensions in 30 secs, with the same Duchon setup, keep the per-axis anisotropy on the Duchon centers enabled and move on to N = 200 in 60s.

Once this is done, move on to N = 1000 in 120s.

Then, after you have reliable and beautiful and correct code fitting ansiotropic Duchon marginal slope in 120s at N=1000, hill-climb it. How much more can you do in 120s? In 60s? Etc.

NOTE: it's important to MONITOR PROGRESS and stop runs early, possibly in even 10s, if it is obvious they will not progress fast enough to meet goals. ITERATE EARLY AND OFTEN. DO NOT let it run to timeout if you can tell faster that it is unlikely to complete the goal.

Once you have sufficiently hill-climbed 120s and 60s, your goal will be to put as MANY CASES AND CONTROLS into 5 minute fit as possible, changing Rust code to enable it.

You can do ~any strategy to achieve your goals. BUT, it must be CORRECT MATH, PRINCIPLED, BEAUTIFUL, PROPER, and DEPLOYABLE. We want the ABSOLUTE BEST long-term solution ALWAYS. We want things FULLY ACHIEVED, WIRED IN, ETC.

If Rust code is compiling, DO NOT wait. Run it in background. Find something else to do. Do DEEP analysis, tasks, etc., not pointless tasks, while this is happening. COMMIT TO LARGE GOALS. DO NOT settle for local optima. Do not do things for the sake of it, even while compiling. YOU MUST remain goal-directed. Do not wait for it.

AFTER you have achieved N = 50,000 fit in 5 minutes, both with and without scale dims, you may add probit linkwiggle. Hillclimb on that. Then score warp as well. Then, if that finishes at N = 30,000 in 5 minutes, bump to N = 100,000 and make that fit as fast as possible. Hillclimb it. Any combination of features should work and be very fast.

Do not do survival model marginal slope. Focus entirely on normal one.

ALWAYS figure out the ROOT CAUSE. DO NOT patch symptoms. NEVER go after symptoms to apply band-aid. We want STRUCTURAL, ROOT CAUSE fixes and improvements. You can take time to search, debug, figure out, determine, what the precise mechanism in the code is for an issue. IF YOU SEE SOMETHING WEIRD OR OFF, this is PERFECT: INVESTIGATE IT! That is VERY useful information that you should use.

What is an example of adressing root cause vs. not root cause?

Problem: code is slow because cost is not progressing but isn't converged.
Good: finding and verifying the mechanism or mistake in the code which leads to the cost failing to progress in the first place.
Bad: adding a cost-stagnation exit so it exits and speeds up.
Worse: relaxing tolerances so it exits even faster.

Why is this bad? It's bad because this software will be deployed in healthcare, and hacks like this have the potential to do real-world harm. Understanding and improving, though, has the potential for great benefit.

Never use until loops or while loops in bash–these are ways to wait. Similarly, do not repeatedly tail a boring file (such as the results of compilation) that you'll be notified about completion on anyway, and do not run ps aux repeatedly–another waiting behavior to avoid. Do not use ps -p or cat with head or tail to check status while something is running, as this is waiting–instead pick the next task.

THANKS!

This is exciting software and will hopefully save lives.

Code notes:

NO HACKS. The user is EXTREMELY concerned about math, model, and code quality, much more so than immediate results. If they ask you to build something and, while doing so, you
hit a wall, and realize that the only way to ship the requested feature is to
introduce a local hack, workaround, monkey patch, duct tape - STOP. STOP
IMMEDIATELLY. Either fix the underlying flaw that blocked you in a ROBUST, WELL
DESIGNED, PRODUCTION READY manner, or be honest that the prompt can't be
completed without hacks.

To make it very clear:

- DO NOT INTRODUCE HACKS IN THE CODEBASE.

- DO NOT COMMIT PARTIAL SOLUTIONS OR WORKAROUNDS.

The author appreciates honesty. Go ahead and update the repo to
provide the necessary support in a well designed, robust way. 

NEVER introduce hacks in the codebase.

Also assume that none of the code you're working in is in production, so,
backwards compatibility is NOT IMPORTANT. If you find something that is poorly
designed and fixing it would require breaking existing APIs or behavior, DO SO.
Do it properly rather than preserving a flawed design. Prioritize clarity,
correctness, and maintainability over compatibility with existing code.

Core values:
- ABSOLUTE code quality over speed of delivery.
- Correctness over convenience.
- Maintainability over short-term productivity.
- Robust design over quick fixes.
- Simplicity over complexity.


This task can always be improved, substantially. Don't give up. Have fun. DO NOT reply with repetitive or ackknowledgements or such. START. DO THE HARD TASK. You can do it.
